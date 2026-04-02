import os
import matplotlib.pyplot as plt


# Replace these lists with your real epoch-wise values.
teacher_train_loss = []
teacher_train_acc = []
teacher_test_acc = []

student_train_loss = []
student_train_acc = []
student_test_acc = []

kd_train_loss = []
kd_train_acc = []
kd_test_acc = []


# Final comparison values from your README.
final_accuracies = {
    "Teacher\n(ResNet-18)": 94.58,
    "Student\n(ResNet-9)": 88.09,
    "KD Student\n(ResNet-9)": 92.51,
}


def plot_metric_curves(epochs, values_dict, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    for label, values in values_dict.items():
        if values:
            plt.plot(epochs[: len(values)], values, marker="o", linewidth=2, label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_final_accuracy_comparison(accuracy_dict, save_path):
    labels = list(accuracy_dict.keys())
    values = list(accuracy_dict.values())

    plt.figure(figsize=(2, 0.5))
    colors = ["#0C355C", "#62261D", "#333913"]
    bars = plt.bar(
        labels,
        values,
        color=colors,
        width=0.22,
        edgecolor="#1A1A1A",
        linewidth=0.8,
    )
    plt.title("Final Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(80, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.gca().set_axisbelow(True)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_color("#444444")
    plt.gca().spines["bottom"].set_color("#444444")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color="#1A1A1A",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)

    max_epochs = max(
        len(teacher_train_loss),
        len(student_train_loss),
        len(kd_train_loss),
        len(teacher_test_acc),
        len(student_test_acc),
        len(kd_test_acc),
        1,
    )
    epochs = list(range(1, max_epochs + 1))

    plot_metric_curves(
        epochs,
        {
            "Teacher Train Loss": teacher_train_loss,
            "Student Train Loss": student_train_loss,
            "KD Student Train Loss": kd_train_loss,
        },
        title="Training Loss Curves",
        ylabel="Loss",
        save_path="figures/loss_curves.png",
    )

    plot_metric_curves(
        epochs,
        {
            "Teacher Train Accuracy": teacher_train_acc,
            "Student Train Accuracy": student_train_acc,
            "KD Student Train Accuracy": kd_train_acc,
        },
        title="Training Accuracy Curves",
        ylabel="Accuracy (%)",
        save_path="figures/train_accuracy_curves.png",
    )

    plot_metric_curves(
        epochs,
        {
            "Teacher Test Accuracy": teacher_test_acc,
            "Student Test Accuracy": student_test_acc,
            "KD Student Test Accuracy": kd_test_acc,
        },
        title="Test Accuracy Curves",
        ylabel="Accuracy (%)",
        save_path="figures/test_accuracy_curves.png",
    )

    plot_final_accuracy_comparison(
        final_accuracies,
        save_path="figures/final_accuracy_comparison.png",
    )

    print("Saved plots to the 'figures' folder.")


if __name__ == "__main__":
    main()
