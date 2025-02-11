from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from typing import Any
import os
import pandas as pd
import uuid
import datetime
from json import dumps
from pydantic import BaseModel
import pytz
import numpy as np
from math import log2
from random import random
from operator import itemgetter

DISTANCE_L2 = "l2"
DISTANCE_COSINE = "cosine"


def l2_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_distance(a, b):
    """ https://stackoverflow.com/questions/58381092/difference-between-cosine-similarity-and-cosine-distance """
    
    return (1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class HNSW:
    """Navigable small world models are defined as any network with
    (poly/)logarithmic complexity using greedy routing.

    The efficiency of greedy routing breaks down for larger networks
    (1-10K+ vertices) when a graph is not navigable [7].
    """

    def __init__(
        self, distance_type, m=5, ef=200, m0=None, heuristic=True, vectorized=False
    ):
        self.data = []
        if distance_type == DISTANCE_L2:
            distance_func = l2_distance
        elif distance_type == DISTANCE_COSINE:
            distance_func = cosine_distance
        else:
            raise TypeError("Please check your distance type!")

        self.distance_func = distance_func

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        # the number of edges per node
        self._m = m
        # The number of neighbors to consider for each node in the index construction
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / np.log2(m)
        self._graphs = []
        self._enter_point = None

        self._select = self._select_heuristic if heuristic else self._select_naive

    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        # level at which the element will be inserted
        level = int(-log2(random()) * self._level_mult) + 1

        # elem will be at data[idx]
        idx = len(data)
        data.append(elem)

        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = distance(elem, data[point])
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            # at these levels we have to insert elem; ep is a heap of entry points.
            ep = [(-dist, point)]
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                ep = self._search_graph(elem, ep, layer, ef)
                # insert in g[idx] the best neighbors
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                # assert len(layer_idx) <= level_m
                # insert backlinks to the new node
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
                    # assert len(g[j]) <= level_m
                # assert all(e in g for _, e in ep)
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: {}})
            self._enter_point = idx

    def balanced_add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        m0 = self._m0

        idx = len(data)
        data.append(elem)

        if point is not None:
            dist = distance(elem, data[point])
            pd = [(point, dist)]
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
                pd.append((point, dist))
            for level, layer in enumerate(graphs):
                level_m = m0 if level == 0 else m
                candidates = self._search_graph(elem, [(-dist, point)], layer, ef)
                layer[idx] = layer_idx = {}
                self._select(layer_idx, candidates, level_m, layer, heap=True)
                # add reverse edges
                for j, dist in layer_idx.items():
                    self._select(layer[j], [idx, dist], level_m, layer)
                    assert len(layer[j]) <= level_m
                if len(layer_idx) < level_m:
                    return
                if level < len(graphs) - 1:
                    if any(p in graphs[level + 1] for p in layer_idx):
                        return
                point, dist = pd.pop()
        graphs.append({idx: {}})
        self._enter_point = idx

    def search(self, q, k=None, ef=None):
        """Find the k points closest to q."""

        distance = self.distance
        graphs = self._graphs
        point = self._enter_point

        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        dist = distance(q, self.data[point])
        # look for the closest neighbor from the top to the 2nd level
        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        # look for ef neighbors in the bottom level
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1(self, q, entry, dist, layer):
        """Equivalent to _search_graph when ef=1."""

        vectorized_distance = self.vectorized_distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = {entry}

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))

        return best, best_dist

    def _search_graph(self, q, ep, layer, ef):
        vectorized_distance = self.vectorized_distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = {p for _, p in ep}

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _select_naive(self, d, to_insert, m, layer, heap=False):
        if not heap:
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)  # smallest m distances
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            assert len(d) == m

    def _select_heuristic(self, d, to_insert, m, g, heap=False):
        nb_dicts = [g[idx] for idx in d]

        def prioritize(idx, dist):
            return any(nd.get(idx, float("inf")) < dist for nd in nb_dicts), dist, idx

        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(
                m, (prioritize(idx, -mdist) for mdist, idx in to_insert)
            )

        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)

        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(
                to_check, (prioritize(idx, dist) for idx, dist in d.items())
            )
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            assert len(d) == m

    def __getitem__(self, idx):
        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return
            

class LazyBucket(BaseModel):
    """This class is a representation of a "lazy" data bucket for a distributed
    search system. The lazy loading technique is used to defer initialization
    of an object until the point at which it is needed.

    Attributes:
        db_location (str): The path to the location where the bucket data is stored.
        segment_index (int): The index of the current segment. This index is used to name the bucket file.
        bucket_name (str, optional): The name of the parquet file where the bucket data is stored.
                                     The segment index will be inserted in place of '{}'. Defaults to "segment-{}.parquet".
        metadata_name (str, optional): The name of the json file where the metadata associated with the bucket is stored.
                                        The segment index will be inserted in place of '{}'. Defaults to "segment-{}-metadata.json".
        loaded (bool, optional): A flag that indicates whether the bucket has been loaded. Defaults to False.
        dirty (bool, optional): A flag that indicates whether there are unsaved changes in the bucket. Defaults to False.
        frame (pandas.DataFrame | None, optional): The pandas DataFrame that contains the data of the bucket. Defaults to None.
        frame_schema (list[str], optional): The schema of the DataFrame. Defaults to ["id", "vector", "metadata", "document", "timestamp"].
        vectors (list): A list to store vectors from the DataFrame for easy access and manipulation. Defaults to an empty list.
        dirty_rows (list): A list to store new rows that haven't been added to the DataFrame yet. Defaults to an empty list.
    """

    db_location: str
    segment_index: str
    bucket_name: str = "segment-{}.parquet"
    metadata_name: str = "segment-{}-metadata.json"
    loaded: bool = False
    dirty: bool = False
    frame: Any | None = None
    frame_schema: str = ["id", "vector", "metadata", "document", "timestamp"]
    vectors = []
    dirty_rows = []
    hnsw: Any = None
    attrs: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hnsw = HNSW("cosine", m0=5, ef=10)

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.segment_index=} {len(self.vectors)=} {self.dirty=} {self.loaded=} )>"

    @property
    def key(self):
        return self.bucket_name.format(self.segment_index)

    @property
    def frame_location(self):
        bucket_name = self.bucket_name.format(self.segment_index)
        return f"{self.db_location}/{bucket_name}"

    def _lazy_load(self):
        if self.loaded:
            return

        if os.path.exists(self.frame_location):
            self.frame = pd.read_parquet(self.frame_location)
        else:
            self.frame = pd.DataFrame(columns=self.frame_schema)
            self.attrs = self.frame.attrs
        if list(self.frame.columns) != self.frame_schema:
            raise ValueError(f"Invalid frame_schema {self.frame.columns=}")
        self.loaded = True
        self.vectors = self.frame["vector"].tolist()
        self.dirty_rows = self.frame.to_dict("records")
        for v in self.vectors:
            self.hnsw.add(v)

    def append(self, vector: np.ndarray, **attrs):
        if not self.loaded:
            self._lazy_load()
        uid = uuid.uuid1().urn
        document = {
            "id": uid,
            "vector": vector,
            "metadata": attrs.get("metadata", {"name": "unknown"}),
            "document": attrs.get("document", ""),
            "timestamp": attrs.get("timestamp", datetime.datetime.now(pytz.UTC)),
        }

        self.dirty_rows.append(document)
        self.dirty = True
        self.vectors.append(vector)
        self.hnsw.add(vector)
        return uid

    def search(self, vector: np.ndarray, k: int = 4):
        self._lazy_load()
        try:
            results = self.hnsw.search(vector, k)
        except ValueError:  # Empty graph
            return []
        return results

    def sync(self, **attrs):
        if not self.dirty:
            return

        self.frame = self.frame._append(self.dirty_rows, ignore_index=True)
        if self.frame.empty:
            return
        # TODO: eval last sync time
        # self.frame.attrs["last_update"] = datetime.datetime.now(pytz.UTC)
        now_dt = datetime.datetime.now(pytz.UTC)
        self.frame.attrs["last_update"] = dumps(now_dt, indent=4, sort_keys=True, default=str)
        for k, v in attrs.items():
            self.frame.attrs[k] = v

        os.makedirs(self.db_location, exist_ok=True)
        self.frame.to_parquet(self.frame_location, engine='pyarrow', compression="gzip")
        self.dirty = False

    def __len__(self):
        if not self.loaded:
            self._lazy_load()
        return len(self.vectors)

    def memory_footprint(self):
        return self.frame.memory_usage(deep=True).sum()





def lambda_handler(event, context):

    print(event)
    lake_name = event['lake_name']
    dimensions = event['dimensions']
    approx_shards = event['approx_shards']
    num_hashes = event['num_hashes']
    bucket_size = event['bucket_size']
    hyperplanes = 13
    shard_index = event['shard_index']
    embedding = event['embedding']
    metadata = {"id": "1"}

    # TODO: instantiate a lazy bucket
    # TODO: append the vector/embedding to the bucket
    # TODO: if the embedding's corresponding segment is in S3, use that segment
    # TODO: if the embedding's corresponding segment isn't in S3, 
    # make the appropriate segment
    # TODO: save the segment to S3 using the convention '{lake_name}/{segment}'




    print("hello world!")