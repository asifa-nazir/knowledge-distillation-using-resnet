# CIFAR-10 Knowledge Distillation

This project compares a stronger teacher model against a smaller student model on CIFAR-10 and then improves the student with knowledge distillation.

## Models

- Teacher: ResNet-18
- Student: ResNet-9
- KD Student: ResNet-9 trained with soft targets from the ResNet-18 teacher

## Project Files

- `dataset.py`: CIFAR-10 dataloaders and data augmentation
- `models.py`: teacher and student model definitions
- `train_teacher.py`: train the teacher model
- `train_student.py`: train the student baseline
- `train_KD.py`: train the distilled student

## Results

<div align="center">
<table>
  <tr>
    <td valign="top">
      <table>
        <tr>
          <th>Model</th>
          <th>Role</th>
          <th>Method</th>
          <th>Acc (%)</th>
        </tr>
        <tr>
          <td>ResNet-18</td>
          <td>Teacher</td>
          <td>Supervised</td>
          <td>94.58</td>
        </tr>
        <tr>
          <td>ResNet-9</td>
          <td>Student</td>
          <td>Supervised</td>
          <td>88.09</td>
        </tr>
        <tr>
          <td>ResNet-9</td>
          <td>KD Student</td>
          <td>KD</td>
          <td>92.51</td>
        </tr>
      </table>
      <sub>
        Trained ResNet-18 and ResNet-9 using supervised learning,<br>
        then distilled knowledge from the ResNet-18 teacher<br>
        into the ResNet-9 student using KD.
      </sub>

  </td>
    <td valign="top" align="center">
      <img src="figures/final_accuracy_comparison.png" alt="Final Accuracy Comparison" width="580">
      <br>
      <sub>Comparison of teacher, student, and KD student accuracy on CIFAR-10.</sub>
    </td>
  </tr>
</table>
</div>

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the teacher:

```bash
python train_teacher.py
```

Train the student baseline:

```bash
python train_student.py
```

Train the distilled student:

```bash
python train_KD.py
```

## Saved Weights

- `kd_student.pth`: distilled ResNet-9 weights created after running `train_KD.py`

## Notes

- Resume checkpoints are ignored by git and are not pushed to GitHub.
- The teacher weights should be saved as `teacher.pth` before running `train_KD.py`.
- The distilled student weights are saved as `kd_student.pth`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
