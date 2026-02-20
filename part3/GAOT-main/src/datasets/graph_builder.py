"""
Graph building utilities for variable coordinate datasets.
Handles neighbor computation and graph construction for VX mode.
"""
import time
import torch
from typing import List, Tuple, Optional, Callable

from ..model.layers.utils.neighbor_search import (
    NeighborSearch,
    compute_knn_radius,
    compute_query_knn_radius
)
from ..utils.scaling import rescale


class GraphBuilder:
    """
    Builds encoder and decoder graphs for variable coordinate datasets.
    Handles neighbor computation with multiple radius scales.
    """
    
    def __init__(self,
                 neighbor_search_method: str = "auto",
                 dynamic_radius: bool = False,
                 dynamic_radius_k: int = 8,
                 dynamic_radius_alpha: float = 1.5,
                 coord_scaler: Optional[Callable] = None):
        """
        Initialize graph builder.
        
        Args:
            neighbor_search_method: Method for neighbor search
        """
        self.nb_search = NeighborSearch(neighbor_search_method)
        self.dynamic_radius = dynamic_radius
        self.dynamic_radius_k = dynamic_radius_k
        self.dynamic_radius_alpha = dynamic_radius_alpha
        self.coord_scaler = coord_scaler
        self.dynamic_nb_search = None
        if self.dynamic_radius and neighbor_search_method not in ["chunked", "native"]:
            self.dynamic_nb_search = NeighborSearch(method="chunked")

    def _neighbor_search(self,
                         data: torch.Tensor,
                         queries: torch.Tensor,
                         radius: torch.Tensor) -> dict:
        if isinstance(radius, torch.Tensor):
            searcher = self.dynamic_nb_search or self.nb_search
            return searcher(data, queries, radius)
        return self.nb_search(data, queries, radius)

    def _invert_neighbors_csr(self,
                              neighbors: dict,
                              num_queries: int,
                              num_data: int) -> dict:
        neighbors_index = neighbors["neighbors_index"].long()
        row_splits = neighbors["neighbors_row_splits"].long()
        device = neighbors_index.device

        if neighbors_index.numel() == 0:
            return {
                "neighbors_index": neighbors_index.new_empty((0,), dtype=torch.long),
                "neighbors_row_splits": torch.zeros(
                    num_data + 1, device=device, dtype=torch.long
                )
            }

        counts = row_splits[1:] - row_splits[:-1]
        query_ids = torch.repeat_interleave(
            torch.arange(num_queries, device=device, dtype=torch.long),
            counts
        )
        order = torch.argsort(neighbors_index)
        phys_sorted = neighbors_index[order]
        query_sorted = query_ids[order]
        counts_phys = torch.bincount(phys_sorted, minlength=num_data)
        row_splits_out = torch.cat([
            torch.tensor([0], device=device, dtype=torch.long),
            torch.cumsum(counts_phys, dim=0)
        ])

        return {
            "neighbors_index": query_sorted,
            "neighbors_row_splits": row_splits_out
        }
    
    def build_graphs_for_split(self, x_data: torch.Tensor, latent_queries: torch.Tensor,
                              gno_radius: float, scales: List[float]) -> Tuple[List, List]:
        """
        Build encoder and decoder graphs for a data split.
        
        Args:
            x_data: Coordinate data [n_samples, n_nodes, coord_dim] or [n_samples, 1, n_nodes, coord_dim]
            latent_queries: Latent query coordinates [n_latent, coord_dim] or
                [n_samples, n_latent, coord_dim] for per-sample tokens
            gno_radius: Base radius for neighbor search
            scales: List of scale factors for multi-scale graphs
            
        Returns:
            tuple: (encoder_graphs_list, decoder_graphs_list)
        """
        print(f"Building graphs for {len(x_data)} samples...")
        start_time = time.time()
        
        encoder_graphs = []
        decoder_graphs = []

        latent_batched = latent_queries.dim() == 3
        if latent_batched and latent_queries.shape[0] != len(x_data):
            raise ValueError(
                "latent_queries batch size must match x_data when using per-sample tokens"
            )

        latent_radius = None
        if self.dynamic_radius and not latent_batched:
            latent_radius = compute_knn_radius(
                latent_queries,
                k=self.dynamic_radius_k,
                alpha=self.dynamic_radius_alpha,
                fallback_radius=gno_radius
            )
        
        for i, x_sample in enumerate(x_data):
            # Handle different input shapes
            if x_sample.dim() == 3 and x_sample.shape[0] == 1:
                # Shape: [1, n_nodes, coord_dim] -> [n_nodes, coord_dim]
                x_coord = x_sample[0]
            elif x_sample.dim() == 2:
                # Shape: [n_nodes, coord_dim]
                x_coord = x_sample
            else:
                raise ValueError(f"Unexpected coordinate shape: {x_sample.shape}")
            
            if self.coord_scaler is not None:
                x_coord_scaled = self.coord_scaler(x_coord)
            else:
                x_coord_scaled = rescale(x_coord, (-1, 1))

            latent_query = latent_queries[i] if latent_batched else latent_queries
            if self.dynamic_radius and latent_batched:
                latent_radius = compute_knn_radius(
                    latent_query,
                    k=self.dynamic_radius_k,
                    alpha=self.dynamic_radius_alpha,
                    fallback_radius=gno_radius
                )
            
            # Build encoder graphs (physical -> latent)
            encoder_nbrs_sample = []
            for scale in scales:
                if self.dynamic_radius:
                    scaled_radius = latent_radius * scale
                else:
                    scaled_radius = gno_radius * scale
                with torch.no_grad():
                    nbrs = self._neighbor_search(x_coord_scaled, latent_query, scaled_radius)
                encoder_nbrs_sample.append(nbrs)
            encoder_graphs.append(encoder_nbrs_sample)
            
            # Build decoder graphs (latent -> physical)
            decoder_nbrs_sample = []
            if self.dynamic_radius:
                use_query_radius = getattr(self, "dynamic_decoder_query_radius", False)
                if use_query_radius:
                    try:
                        decoder_radius = compute_query_knn_radius(
                            x_coord_scaled,
                            latent_query,
                            k=self.dynamic_radius_k,
                            alpha=self.dynamic_radius_alpha,
                            fallback_radius=gno_radius
                        )
                        for scale in scales:
                            scaled_radius = decoder_radius * scale
                            with torch.no_grad():
                                nbrs = self._neighbor_search(latent_query, x_coord_scaled, scaled_radius)
                            decoder_nbrs_sample.append(nbrs)
                    except Exception as exc:
                        print(f"Falling back to decoder graph inversion: {exc}")
                        decoder_nbrs_sample = []
                        num_latent = latent_query.shape[0]
                        num_nodes = x_coord_scaled.shape[0]
                        for scale_idx in range(len(scales)):
                            encoder_nbrs_scale = encoder_nbrs_sample[scale_idx]
                            decoder_nbrs_sample.append(
                                self._invert_neighbors_csr(
                                    encoder_nbrs_scale,
                                    num_queries=num_latent,
                                    num_data=num_nodes
                                )
                            )
                else:
                    num_latent = latent_query.shape[0]
                    num_nodes = x_coord_scaled.shape[0]
                    for scale_idx in range(len(scales)):
                        encoder_nbrs_scale = encoder_nbrs_sample[scale_idx]
                        decoder_nbrs_sample.append(
                            self._invert_neighbors_csr(
                                encoder_nbrs_scale,
                                num_queries=num_latent,
                                num_data=num_nodes
                            )
                        )
            else:
                for scale in scales:
                    scaled_radius = gno_radius * scale
                    with torch.no_grad():
                        nbrs = self._neighbor_search(latent_query, x_coord_scaled, scaled_radius)
                    decoder_nbrs_sample.append(nbrs)
            decoder_graphs.append(decoder_nbrs_sample)
            
            if (i + 1) % 100 == 0 or i == len(x_data) - 1:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{len(x_data)} samples ({elapsed:.2f}s)")
        
        total_time = time.time() - start_time
        print(f"Graph building completed in {total_time:.2f}s")
        
        return encoder_graphs, decoder_graphs
    
    def build_all_graphs(self, data_splits: dict, latent_queries: torch.Tensor,
                        gno_radius: float, scales: List[float],
                        build_train: bool = True) -> dict:
        """
        Build graphs for all data splits.
        
        Args:
            data_splits: Dictionary with train/val/test splits
            latent_queries: Latent query coordinates
            gno_radius: Base radius for neighbor search
            scales: Scale factors for multi-scale graphs
            build_train: Whether to build train/val graphs (skip if testing only)
            
        Returns:
            dict: Dictionary with encoder/decoder graphs for each split
        """
        all_graphs = {}
        
        def _select_latent(split_name: str):
            if isinstance(latent_queries, dict):
                if split_name not in latent_queries:
                    raise KeyError(f"latent_queries missing split '{split_name}'")
                return latent_queries[split_name]
            return latent_queries

        # Always build test graphs
        if 'test' in data_splits:
            print("Building test graphs...")
            latent_test = _select_latent('test')
            encoder_test, decoder_test = self.build_graphs_for_split(
                data_splits['test']['x'], latent_test, gno_radius, scales
            )
            all_graphs['test'] = {
                'encoder': encoder_test,
                'decoder': decoder_test
            }
        
        # Build train/val graphs if requested
        if build_train:
            if 'train' in data_splits:
                print("Building train graphs...")
                latent_train = _select_latent('train')
                encoder_train, decoder_train = self.build_graphs_for_split(
                    data_splits['train']['x'], latent_train, gno_radius, scales
                )
                all_graphs['train'] = {
                    'encoder': encoder_train,
                    'decoder': decoder_train
                }
            
            if 'val' in data_splits:
                print("Building val graphs...")
                latent_val = _select_latent('val')
                encoder_val, decoder_val = self.build_graphs_for_split(
                    data_splits['val']['x'], latent_val, gno_radius, scales
                )
                all_graphs['val'] = {
                    'encoder': encoder_val,
                    'decoder': decoder_val
                }
        else:
            print("Skipping train/val graph building (testing mode)")
            all_graphs['train'] = None
            all_graphs['val'] = None
        
        return all_graphs
    
    def validate_graphs(self, graphs: dict, expected_samples: dict):
        """
        Validate that graphs have correct structure and sizes.
        
        Args:
            graphs: Graph dictionary
            expected_samples: Expected number of samples per split
        """
        for split_name, split_graphs in graphs.items():
            if split_graphs is None:
                continue
                
            encoder_graphs = split_graphs['encoder']
            decoder_graphs = split_graphs['decoder']
            expected_count = expected_samples.get(split_name, 0)
            
            assert len(encoder_graphs) == expected_count, \
                f"Encoder graphs for {split_name}: expected {expected_count}, got {len(encoder_graphs)}"
            assert len(decoder_graphs) == expected_count, \
                f"Decoder graphs for {split_name}: expected {expected_count}, got {len(decoder_graphs)}"
            
            # Validate individual samples
            for i, (enc_sample, dec_sample) in enumerate(zip(encoder_graphs, decoder_graphs)):
                assert isinstance(enc_sample, list), f"Encoder sample {i} should be list of scales"
                assert isinstance(dec_sample, list), f"Decoder sample {i} should be list of scales"
                assert len(enc_sample) == len(dec_sample), \
                    f"Encoder and decoder should have same number of scales for sample {i}"
        
        print("Graph validation passed")


class CachedGraphBuilder(GraphBuilder):
    """
    Graph builder with caching capabilities.
    Can save and load pre-computed graphs to avoid recomputation.
    """
    
    def __init__(self,
                 neighbor_search_method: str = "auto",
                 cache_dir: Optional[str] = None,
                 dynamic_radius: bool = False,
                 dynamic_radius_k: int = 8,
                 dynamic_radius_alpha: float = 1.5,
                 coord_scaler: Optional[Callable] = None):
        super().__init__(
            neighbor_search_method=neighbor_search_method,
            dynamic_radius=dynamic_radius,
            dynamic_radius_k=dynamic_radius_k,
            dynamic_radius_alpha=dynamic_radius_alpha,
            coord_scaler=coord_scaler
        )
        self.cache_dir = cache_dir
    
    def _get_cache_path(self, dataset_name: str, split_name: str, graph_type: str) -> str:
        """Get cache file path for graphs."""
        if self.cache_dir is None:
            raise ValueError("Cache directory not specified")
        
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{dataset_name}_{split_name}_{graph_type}_graphs.pt")
    
    def save_graphs(self, graphs: dict, dataset_name: str):
        """Save graphs to cache."""
        if self.cache_dir is None:
            print("No cache directory specified, skipping graph save")
            return
        
        for split_name, split_graphs in graphs.items():
            if split_graphs is None:
                continue
            
            # Save encoder graphs
            encoder_path = self._get_cache_path(dataset_name, split_name, "encoder")
            torch.save(split_graphs['encoder'], encoder_path)
            
            # Save decoder graphs
            decoder_path = self._get_cache_path(dataset_name, split_name, "decoder")
            torch.save(split_graphs['decoder'], decoder_path)
        
        print(f"Graphs saved to cache directory: {self.cache_dir}")
    
    def load_graphs(self, dataset_name: str, splits: List[str]) -> Optional[dict]:
        """Load graphs from cache."""
        if self.cache_dir is None:
            return None
        
        try:
            all_graphs = {}
            for split_name in splits:
                encoder_path = self._get_cache_path(dataset_name, split_name, "encoder")
                decoder_path = self._get_cache_path(dataset_name, split_name, "decoder")
                
                import os
                if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
                    print(f"Cache files not found for split: {split_name}")
                    return None
                
                encoder_graphs = torch.load(encoder_path)
                decoder_graphs = torch.load(decoder_path)
                
                all_graphs[split_name] = {
                    'encoder': encoder_graphs,
                    'decoder': decoder_graphs
                }
            
            print(f"Graphs loaded from cache directory: {self.cache_dir}")
            return all_graphs
            
        except Exception as e:
            print(f"Failed to load graphs from cache: {e}")
            return None
    
    def build_all_graphs(self, data_splits: dict, latent_queries: torch.Tensor,
                        gno_radius: float, scales: List[float],
                        dataset_name: str = "dataset", build_train: bool = True,
                        use_cache: bool = True) -> dict:
        """
        Build graphs with caching support.
        
        Args:
            data_splits: Data splits dictionary
            latent_queries: Latent query coordinates
            gno_radius: Base radius
            scales: Scale factors
            dataset_name: Name for cache files
            build_train: Whether to build train/val graphs
            use_cache: Whether to use cached graphs
            
        Returns:
            dict: Graph dictionary
        """
        # Try to load from cache first
        if use_cache and self.cache_dir is not None:
            cache_splits = ['test']
            if build_train:
                cache_splits.extend(['train', 'val'])
            
            cached_graphs = self.load_graphs(dataset_name, cache_splits)
            if cached_graphs is not None:
                return cached_graphs
        
        # Build graphs if not cached
        graphs = super().build_all_graphs(
            data_splits, latent_queries, gno_radius, scales, build_train
        )
        
        # Save to cache
        if use_cache:
            self.save_graphs(graphs, dataset_name)
        
        return graphs
