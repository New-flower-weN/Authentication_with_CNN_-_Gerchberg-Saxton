import sys
sys.path.append('..') # adding root path of project for module search
from src.hologram_generator import HologramGenerator
from config.config import N, ps, lmbd, z

def main():
    generator = HologramGenerator(N=N, ps=ps, lmbd=lmbd, z=z)
    
    generator.save_initial_phase('./data/dataset_32/phase_1.png')
    
    start_idx = 100001
    end_idx = 170001
    output_dir = './data/dataset_32'
    
    generator.generate_dataset(start_idx, end_idx, output_dir)

if __name__ == "__main__":
    main()