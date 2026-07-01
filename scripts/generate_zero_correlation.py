import sys
sys.path.append('..') # adding root path of project for module search
from src.zero_correlation_generator import ZeroCorrelationGenerator
from config.config import N, ps, lmbd, z

def main():
    generator = ZeroCorrelationGenerator(N=N, ps=ps, lmbd=lmbd, z=z)
    
    start_idx = 100001
    end_idx = 160001
    input_dir = './data/dataset_32'
    output_dir = './data/dataset_32'
    
    generator.generate_dataset(start_idx, end_idx, input_dir, output_dir)

if __name__ == "__main__":
    main()