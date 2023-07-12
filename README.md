# Mitigating Negative Transfer with Task Awareness for Sexism, Hate Speech, and Toxic Language Detection


<img src=".\Figures\MTL-models.svg">

# Description
This repository contains the code for the two proposed models in the paper **Mitigating Negative Transfer with Task Awareness for Sexism, Hate Speech, and Toxic Language Detection**. 
This paper will be published in the proceddings of the [2023 International Joint Conference on Neural Networks (IJCNN)](https://2023.ijcnn.org/). Descriptions of the implementation and the dataset are contained in the [paper](https://arxiv.org/pdf/2307.03377.pdf).

# Paper Abstract
This paper proposes a novelty approach to mitigate the negative transfer problem. In the field of machine learning, the common strategy is to apply the Single-Task Learning approach in order to train a supervised model to solve a specific task. Training a robust model requires a lot of data and a significant amount of computational resources, making this solution unfeasible in cases where data are unavailable or expensive to gather. Therefore another solution, based on the sharing of information between tasks, has been developed: Multi-task Learning (MTL). Despite the recent developments regarding MTL, the problem of negative transfer has still to be solved. It is a phenomenon that occurs when noisy information is shared between tasks, resulting in a drop in performance. This paper proposes a new approach to mitigate the negative transfer problem based on the task awareness concept. The proposed approach results in diminishing the negative transfer together with an improvement of performance over classic MTL solution. Moreover, the proposed approach has been implemented in two unified architectures to detect Sexism, Hate Speech, and Toxic Language in text comments. The proposed architectures set a new state-of-the-art both in EXIST-2021 and HatEval-2019 benchmarks.

# Cite
If you find this [article](https://arxiv.org/pdf/2307.03377.pdf) or the [code](https://github.com/AngelFelipeMP/Mitigating-Negative-Transfer-with-Task-Awareness) useful in your research, please cite us as:

```
@inproceedings{depaula2022mitigati,
  title={Mitigating Negative Transfer with Task Awareness for Sexism, Hate Speech, and Toxic Language Detection},
  author={Magnoss{\~a}o de Paula, Angel Felipe and Rosso, Paolo and Spina, Damiano},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)},
  year={2023},
  organization={IEEE},
  address={Gold Coast, Australia}
}
```

# Credits
IJCNN 2023 Organizers

IJCNN 2023 proceedings: (link coming soon)

Conference website: https://2023.ijcnn.org/

# Acknowledgments
The work of Paolo Rosso was in the framework of
the FairTransNLP-Stereotypes research project (PID2021-
124361OB-C31) on Fairness and Transparency for equitable
NLP applications in social media: Identifying stereotypes
and prejudices and developing equitable systems, funded by
MCIN/AEI/10.13039/501100011033 and by ERDF, EU A way
of making Europe. Damiano Spina is the recipient of an
Australian Research Council DECRA Research Fellowship
(DE200100064).
