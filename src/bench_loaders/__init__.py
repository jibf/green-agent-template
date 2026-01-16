from .base_loader import BaseLoader
from .tau_bench_loader import TauBenchLoader
from .tau2_bench_loader import Tau2BenchLoader
from .ace_bench_loader import AceBenchLoader
from .complex_func_bench_loader import ComplexFuncBenchLoader
from .drafter_bench_loader import DrafterBenchLoader
from .bfcl_loader import BfclLoader

from ..utils.types import Benchmark

def get_bench_loader(benchmark: Benchmark):
    """Get the appropriate loader class for the given benchmark."""
    loader_map = {
        Benchmark.TAU_BENCH: TauBenchLoader,
        Benchmark.TAU2_BENCH: Tau2BenchLoader,
        Benchmark.COMPLEX_FUNC_BENCH: ComplexFuncBenchLoader,
        Benchmark.DRAFTER_BENCH: DrafterBenchLoader,
        Benchmark.ACE_BENCH: AceBenchLoader,
        Benchmark.BFCL: BfclLoader,
    }
    
    if benchmark not in loader_map:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    
    return loader_map[benchmark]
