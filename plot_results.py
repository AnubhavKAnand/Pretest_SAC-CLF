import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

def generate_btp_plots(csv_path="btp_results.csv"):
    df = pd.read_csv(csv_path)
    
    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('HSW-SF Pretest Simulation Results', fontsize=16, fontweight='bold')

    # Plot 1: SAC Convergence (Reward)
    axs[0, 0].plot(df['Episode'], df['Return'], color='blue', linewidth=2)
    axs[0, 0].set_title('SAC Agent Convergence')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')

    # Plot 2: Energy vs Latency Tradeoff
    ax2 = axs[0, 1]
    ax2.plot(df['Episode'], df['Avg_Battery'], color='green', label='Battery Remaining (%)')
    ax2.set_ylabel('Battery (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['Episode'], df['Avg_Queue'], color='red', linestyle='--', label='Avg Queue (Latency)')
    ax2_twin.set_ylabel('Queue Length (Packets)', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    axs[0, 1].set_title('Energy vs. Latency Optimization')
    axs[0, 1].set_xlabel('Episode')

    # Plot 3: The CLF Gatekeeper (Topology Stability)
    ax3 = axs[1, 0]
    ax3.plot(df['Episode'], df['Avg_V_top'], color='purple', label='Topology Error (V_top)')
    ax3.set_ylabel('V_top', color='purple')
    
    ax3_twin = ax3.twinx()
    # Log scale for Eta because it spikes massively during jamming/failure
    ax3_twin.plot(df['Episode'], df['Max_Eta'], color='orange', alpha=0.6, label='Constraint Spikes (Eta)')
    ax3_twin.set_yscale('log')
    ax3_twin.set_ylabel('Constraint Strength (Eta) - Log Scale', color='orange')
    axs[1, 0].set_title('CLF Stability Enforcement')
    axs[1, 0].set_xlabel('Episode')

    # Plot 4: Knowledge Distillation Loss
    axs[1, 1].plot(df['Episode'], df['S_Loss'], color='magenta', label='Student Loss')
    axs[1, 1].plot(df['Episode'], df['T_Loss'], color='cyan', linestyle='-.', label='Teacher Loss (SplitFed)')
    axs[1, 1].set_title('EuroSAT Distillation Learning Curve')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Cross-Entropy Loss')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig("BTP_Results_Dashboard.png", dpi=300)
    print("Saved publication-ready graph to BTP_Results_Dashboard.png!")
    plt.show()

if __name__ == "__main__":
    generate_btp_plots()