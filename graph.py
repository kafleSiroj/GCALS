
import matplotlib.pyplot as plt

def plot(num_epochs, gcals, static, dataset):

    gcals_train, gcals_test, gcals_acc, gcals_label_smoothing = gcals
    static_train, static_test, static_acc, static_label_smoothing = static


    epochs_range = range(1, num_epochs + 1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    #Train Curve
    ax1.plot(epochs_range, gcals_train, color='blue', linewidth=2.5, label='GCALS')
    ax1.plot(epochs_range, static_train, color='orange', linestyle='--', linewidth=2.0, label=f'Static LS (ε = {static_label_smoothing[0]})')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Test Curve
    ax2.plot(epochs_range, gcals_test, color='blue', linewidth=2.5, label=f'GCALS (max_capacity = {static_label_smoothing[0]})')
    ax2.plot(epochs_range, static_test, color='orange', linestyle='--', linewidth=2.0, label=f'Static LS (ε = {static_label_smoothing[0]})')
    ax2.set_title('Test Loss', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss Value')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    # Label Smoothing Curve
    ax3.plot(epochs_range, gcals_label_smoothing, color='red', linestyle='-.', linewidth=2.5, label=f'GCALS max_capacity = {static_label_smoothing[0]} ')
    ax3.plot(epochs_range, static_label_smoothing, color='blue', linestyle=':', linewidth=1.5, label=f'Static Label Smooting = {static_label_smoothing[0]}')
    ax3.set_title('Label Smoothing Parameter (ε)', fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Label Smoothing Parameter')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend()

    # Test Accuracy Curve
    ax4.plot(epochs_range, gcals_acc, color='blue', linewidth=2.5, label=f"GCALS Max: {max(gcals_acc):.2f}%")
    ax4.plot(epochs_range, static_acc, color='orange', linestyle='--', linewidth=2.0, label=f"Static Max: {max(static_acc):.2f}%")
    ax4.set_title('Test Accuracy', fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower right')

    plt.suptitle(f'Resnet18 Optimization Analysis ({dataset.upper()} Dataset) with AdamW', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'gcals_vs_static_{dataset}', dpi=300)
    plt.show()
