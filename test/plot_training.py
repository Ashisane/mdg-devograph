import re
import os
import matplotlib.pyplot as plt

def parse_training_log(log_path):
    data = {"iter": [], "loss": [], "best_loss": [], "w": [], "a": []}
    
    if not os.path.exists(log_path):
        print(f"Error: Could not find {log_path}")
        return data

    with open(log_path, 'r', encoding='utf-16le', errors='ignore') as f:
        text = f.read()

    lines = text.split('\n')
    for line in lines:
        if '] loss=' in line:
            match_iter = re.search(r'\[\s*(\d+)/\d+\]', line)
            match_loss = re.search(r'loss=([\d\.]+)', line)
            match_best = re.search(r'best=([\d\.]+)', line)
            match_w = re.search(r'w=([\d\.]+)', line)
            match_a = re.search(r'a=([\d\.]+)', line)
            
            if match_iter and match_loss:
                data["iter"].append(int(match_iter.group(1)))
                data["loss"].append(float(match_loss.group(1)))
                
                if match_best:
                    data["best_loss"].append(float(match_best.group(1)))
                if match_w:
                    data["w"].append(float(match_w.group(1)))
                if match_a:
                    data["a"].append(float(match_a.group(1)))
                
    return data

def create_visualizations(data, output_path):
    if not data["iter"]:
        print("No training data parsed. Visualization aborted.")
        return

    # Set up a beautiful plot
    plt.style.use('dark_background')  # for a sleek look
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Biological Parameter Learning (Gradient Descent)', fontsize=16, fontweight='bold', color='white')

    # Top Plot - Loss Convergence
    ax1.plot(data["iter"], data["loss"], color='#ff6b6b', alpha=0.6, label='Current Loss', marker='o', markersize=3)
    
    if data["best_loss"]:
        ax1.plot(data["iter"], data["best_loss"], color='#4ecdc4', linewidth=3, label='Best Loss')
    
    ax1.set_ylabel('Energy / Loss Function', fontsize=12)
    ax1.set_title('Objective Function Minimization over Iterations', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right')

    # Bottom Plot - Parameters (w and a)
    if data["w"]:
        ax2.plot(data["iter"], data["w"], color='#ffe66d', linewidth=2.5, label='Adhesion Strength (w)')
    if data["a"]:
        ax2.plot(data["iter"], data["a"], color='#c7f464', linewidth=2.5, linestyle='--', label='Cortical Pull (alpha)')

    ax2.set_xlabel('Training Iteration', fontsize=12)
    ax2.set_ylabel('Parameter Value', fontsize=12)
    ax2.set_title('Evolution of Physical Parameters', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Visualization perfectly saved to {output_path}")

if __name__ == "__main__":
    log_file_path = os.path.join("results", "sim_output_v2.txt")
    output_image_path = os.path.join("results", "training_visualization.png")
    
    training_data = parse_training_log(log_file_path)
    create_visualizations(training_data, output_image_path)
