# KokoroChat: Japanese Psychological Counseling Dialogue Dataset

**KokoroChat** is the largest human-collected Japanese psychological counseling dialogue dataset to date. It was created through role-playing between trained counselors and includes rich, long-form dialogues and detailed client feedback on counseling quality. The dataset supports research on empathetic response generation, dialogue evaluation, and mental health-oriented language modeling.

## 🌟 Key Features

- 6,589 dialogues, collected between 2020 and 2024
- Avg. 91.2 utterances per dialogue
- 480 trained participants simulating real counseling sessions
- 20-dimension Likert-scale client feedback for every session
- Broad topic coverage: mental health, school, family, workplace, romantic issues, etc.

## 📊 Dataset Statistics

| Metric                         | Value     |
|-------------------------------|-----------|
| Dialogues                     | 6,589     |
| Total Utterances              | 600,939   |
| Avg. Utterances per Dialogue  | 91.20     |
| Avg. Length (Counselor)       | 35.84 chars |
| Avg. Length (Client)          | 20.63 chars |
| Evaluation Items per Dialogue | 20        |
| Feedback Score Mean           | 63.58     |
| Language                      | Japanese  |

## 📁 Dataset Structure

Each sample contains:
- A full counseling dialogue with roles labeled (`counselor` / `client`)
- Timestamped utterances
- Structured client feedback on 20 dimensions
- Flags for ethical concern checks (optional)

## 📄 Citation

If you use this dataset, please cite the following paper:

```bibtex
@inproceedings{qi2025kokorochat,
  title     = {KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors},
  author    = {Zhiyang Qi and Takumasa Kaneko and Keiko Takamizo and Mariko Ukiyo and Michimasa Inaba},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  year      = {2025},
  url       = {https://github.com/UEC-InabaLab/KokoroChat}
}
```

## ⚖️ License

KokoroChat is released under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license.

This means:

- ✅ **Free to share** — You may copy and redistribute the material in any medium or format.
- ❌ **No commercial use** — You may not use the material for commercial purposes.
- ❌ **No modifications** — If you remix, transform, or build upon the material, you may not distribute the modified material.

You must give appropriate **credit**, provide a link to this license, and **not suggest endorsement** by the authors.

📄 [View the full license text](https://creativecommons.org/licenses/by-nc-nd/4.0/)
