# Source
### Investigation of Material and Shape Quantities and Their Bias Impact on Sim-to-Real Instance Segmentation Accuracy

Frage zu denen Fakten benötigt werden:
- Viele Daten sind wichtig für Instanzsegmentierung (Allgemin Wissen oder gibt es ein Paper?)
- Bin Picking ist sehr wichtig in der Industrie
- Viele Daten beruhen auf synthetischen Daten
- Meist Tiefenbilder
- Neuere Forschungen deuten auf eine mindestends genauso hohe Accuracy mit RGB Bildern
- In anderen Bereichen wurde schon ein Textur Bias von DeepLearning Modellen bei Objektklassifikation diagnostiziert
- Bisher wurde dieser Bias aber nicht bei der Segmentation durchgeführt und könnte einen bei der Datengenerierung helfen (gegen den Bias vorzugehen?)
- CNN Based Methods are widely used for Instance Segmentation
- Importance of Instance Segmentation
- Einfluss von der Bildgröße auf die Accuracy bei Segmentierung
- Einfluss von der Datensatzgröße auf die Accuracy bei Segmentierung/Deep Learning


---
### Bias

---

### [Can Biases in ImageNet Models Explain Generalization?](https://openaccess.thecvf.com/content/CVPR2024/papers/Gavrikov_Can_Biases_in_ImageNet_Models_Explain_Generalization_CVPR_2024_paper.pdf)

This paper examines the relationship between various biases—such as shape bias, spectral bias, and critical band bias—and the generalization capabilities of neural networks, specifically in image classification. Generalization refers to how well models perform on unseen data, both from within the distribution (rare, long-tail samples) and out-of-distribution (OOD) samples. The study critiques previous works that found correlations between these biases and generalization, proposing that while these biases might correlate with certain benchmarks or training modes, they fail to provide a holistic explanation for generalization across a broad range of conditions.

The researchers conducted a large-scale study using 48 ImageNet-trained models, all with the same architecture (ResNet-50) but trained with different methods. The results showed that biases like shape and spectral biases were insufficient to fully predict generalization performance. For example, adversarially-trained models exhibited stronger correlations with certain biases (like shape or low-frequency biases), but these findings did not generalize to the wider range of models.

Interestingly, some counterintuitive trends were observed, such as models with a strong texture bias performing better on in-distribution data, and models with a high-frequency bias improving generalization (though not adversarial robustness), despite high-frequency features being less discriminative. This highlights that no single bias can explain generalization in deep neural networks, but some biases, like a combination of low-bandwidth and high-frequency biases, might be necessary for achieving better generalization.

The paper concludes that generalization is too complex to be explained by any one bias alone. Future studies are needed to explore combinations of biases and other factors that could lead to better generalization, as well as improved benchmarks to better assess these relationships.

-> One Bias should not "destroys" the ability of generalizing (in case of ImageNet-trained and classification). So there is no need for redesigning the traindata, so that the bias is maybe reduced.


### [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231)

This paper examines how Convolutional Neural Networks (CNNs) recognize objects, traditionally thought to rely on object shapes, but recent studies suggest a stronger reliance on textures. The authors tested this by evaluating CNNs and humans on images where texture and shape cues conflicted. They found that CNNs, especially those trained on ImageNet, are biased towards recognizing textures over shapes, which differs from human behavior. However, by training the same architecture (ResNet-50) on a stylized version of ImageNet that emphasizes shape over texture, the model shifted to a shape-based recognition strategy, aligning better with human visual performance. This shift also improved object detection and robustness to image distortions. The study highlights the importance of shape-based representations in neural networks for more human-like visual recognition and offers insights into CNN biases and their implications for future research.

-> CNN's are biased towards known Texture and stylizing the traindata improves/shifts this bias (in case of Classification with ImageNet-trained)
-> In ImageNet classification task, CNNs use local texture as the preliminary cue, while humans extract global shape information


### [The Origins and Prevalence of Texture Bias in Convolutional Neural Networks](https://arxiv.org/pdf/1911.09071)

The paper investigates the texture bias observed in ImageNet-trained Convolutional Neural Networks (CNNs), revealing that while CNNs typically classify images based on texture rather than shape, they can be trained to recognize shapes effectively. Key findings include:

- Influence of Training Objectives: Different unsupervised training methods and architectures have minimal but significant impacts on texture bias levels.
- Role of Data Augmentation: Employing less aggressive random cropping and adding naturalistic augmentations (like color distortion, noise, and blur) significantly reduces texture bias, enabling models to classify ambiguous images by shape more accurately.
- Outperformance on Test Sets: Models that favor shape classification surpass baseline performance on out-of-distribution test sets.
- Comparison to Human Vision: The differences between CNNs and human vision are attributed more to the training data statistics than to the model architectures or objectives.

The findings emphasize the need to adjust data statistics through augmentation to align CNNs' processing closer to human vision while acknowledging potential pitfalls in over-constraining models to mimic human judgments, which may include inherent biases.

This research contributes to understanding the divergences between human and machine vision and suggests methods for developing more robust and interpretable computer vision systems.

-> 


### [Shape-biased CNNs are Not Always Superior in Out-of-Distribution Robustness](https://openaccess.thecvf.com/content/WACV2024/papers/Qiu_Shape-Biased_CNNs_Are_Not_Always_Superior_in_Out-of-Distribution_Robustness_WACV_2024_paper.pdf)

The paper investigates the relationship between shape and texture information in achieving Out-of-Distribution (OOD) robustness in Deep Learning, particularly in shape-biased Convolutional Neural Networks (CNNs). While it's believed that shape-biased models would generally perform better due to their alignment with human cognitive decision-making, the authors find that shape information isn't always superior in distinguishing categories like animals and that shape-biased models don't consistently outperform others in various OOD scenarios, such as elastic transformations and blurs.

To address these findings, the authors propose a novel method called Shape-Texture Adaptive Recombination (STAR). This method employs a category-balanced dataset to pretrain a debiased backbone and three specialized heads for extracting shape, texture, and debiased features. An instance-adaptive recombination head is then trained to adjust the contributions of these features dynamically for each instance.

The experiments demonstrate that STAR achieves state-of-the-art OOD robustness across several scenarios, including image corruption, adversarial attacks, style shifts, and dataset shifts, validating the effectiveness of the proposed approach in enhancing robustness against diverse challenges.

Overall, the research underscores the complexity of leveraging shape and texture information for improving model performance and suggests that a more nuanced approach, like STAR, can significantly enhance OOD robustness in CNNs.

-> Improving Method to create a dataset, which leads to a stable debiased CNN base method
-> Shape-Bias is not always wanted / the best way for good generalization (unknown data)


### Remove Texture Bias in CNN
- [Deep convolutional networks do not classify based on global object shape](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006613)
- [Informative Dropout for Robust Representation Learning: A Shape-bias Perspective](https://arxiv.org/pdf/2008.04254)
- [Shortcut Learning in Deep Neural Networks](https://arxiv.org/pdf/2004.07780) -> [it's already here](#shortcut-learning-in-deep-neural-networks)
- [Learning Robust Shape-Based Features for Domain Generalization](https://ieeexplore.ieee.org/abstract/document/9050708)


### Shape-Bias leads to robust CNN's
- [Towards Shape Biased Unsupervised Representation Learning for Domain Generalization](https://arxiv.org/pdf/1909.08245)
- [The shape and simplicity biases of adversarially robust ImageNet-trained CNNs](https://arxiv.org/pdf/2006.09373)
- [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231) -> [see here](#imagenet-trained-cnns-are-biased-towards-texture-increasing-shape-bias-improves-accuracy-and-robustness)
- [Delving Deep into the Generalization of Vision Transformers under Distribution Shifts](https://arxiv.org/pdf/2106.07617)
- [Informative Dropout for Robust Representation Learning: A Shape-bias Perspective](https://arxiv.org/pdf/2008.04254)
- [Learning Robust Shape-Based Features for Domain Generalization](https://ieeexplore.ieee.org/abstract/document/9050708)

---
### General

---

### [Shortcut Learning in Deep Neural Networks](https://arxiv.org/pdf/2004.07780)

->  CNNs learn short-cut decision rules based on training data statistics ratherthan capturing intrinsic true decision rules


