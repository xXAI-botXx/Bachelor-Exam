# Sources of Knowledge

### Depth Data and Shape-Texture Biases in Instance Segmentation

Search sources for:
- A lot of data is important for instance segmentation (general knowledge or is there a paper?)
- Bin picking is very important in the industry
- A lot of data is based on synthetic data
- Mostly depth images
- Recent research suggests at least as high accuracy with RGB images
- In other areas, a texture bias of deep learning models has already been diagnosed in object classification
- So far, however, this bias has not been applied to segmentation and could help in data generation (to counteract the bias?)
- CNN Based Methods are widely used for Instance Segmentation
- Importance of Instance Segmentation
- Shape and Texture Bias in Depth Data (+ RGB)
- RGB and Depth data for Instance Segmentation
- Mask RCNN
- Unreal Engine for datasets
- Synthetic data
- Sim-to-real
- Texture bias for instance segmentation
- Data augmentation impact on accuracy
- 3D models and materials (textures) -> influence on accuracy, investigations on number in deep learning applications


There are Following Tags for which you can search:
- TAG_INSTANCE_SEGMENTATION
- TAG_BIN_PICKING
- TAG_MASKRCNN
- TAG_SYNTHETIC_DATA
- TAG_DEPTH_DATA
- TAG_BIAS
- TAG_NOVEL_DATA     (Unseen Data)
- TAG_GENERALIZATION
- TAG_OOD     (Out of Distribution)
- TAG_SIM_TO_REAL
- TAG_CNN
- TAG_DATA_AUGMENTATION
- TAG_DATA_AMOUNT

- TAG_USEFUL
- TAG_NOT_SO_USEFUL


Additionally the papers are ordered after topics, but a paper often should be in multiple topics and so I recommend you to use the tags.


Tasks:
- update Tags
- find more papers
- summarize papers
- Upload papers and bachelor documents in onedrive



Core-Papers for your work:
- [Trapped in Texture Bias? A Large Scale Comparison of Deep Instance Segmentation](#trapped-in-texture-bias-a-large-scale-comparison-of-deep-instance-segmentation)
- [Can Biases in ImageNet Models Explain Generalization?](#can-biases-in-imagenet-models-explain-generalization) ?
- [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](#imagenet-trained-cnns-are-biased-towards-texture-increasing-shape-bias-improves-accuracy-and-robustness)
- [Towards Synthetic Data: Dealing with the Texture-Bias in Sim2real Learning](#towards-synthetic-data-dealing-with-the-texture-bias-in-sim2real-learning) ?
- ...

Also ver interesting:
- [A Competition of Shape and Texture Bias by Multi-view Image Representation](#a-competition-of-shape-and-texture-bias-by-multi-view-image-representation)

FIXME




---
# CNN

---

### [Shortcut Learning in Deep Neural Networks](https://arxiv.org/pdf/2004.07780)

TAG_GENERALIZATION, TAG_BIAS, TAG_CNN

->  CNNs learn short-cut decision rules based on training data statistics ratherthan capturing intrinsic true decision rules
-> Removing Texture Bias (?)
-> the difference between their architecture and human visual system has been broadly researched


### [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

TAG_GENERALIZATION, TAG_CNN

- deep convolutional models have shown an outstanding performance in various perception tasks, such as image classification, object detection and segmentation. Still, it is not yet quite clear how CNNs reach their decisions


### [Local features and global shape information in object classification by deep convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S0042698920300638)

TAG_CNN

- recent studies have shown that CNNs differ from humans in many crucial aspects


### [Deep Networks Can Resemble Human Feed-forward Vision in Invariant Object Recognition](https://arxiv.org/abs/1508.03929#)

TAG_CNN

- the difference between their architecture and human visual system has been broadly researched


### [Deep convolutional networks do not classify based on global object shape](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006613)

TAG_CNN, TAG_BIAS

- DCNNs and Shape Recognition: Deep convolutional neural networks (DCNNs) struggle to recognize global object shapes, unlike humans who rely heavily on abstract, global shape information for recognition across varying conditions.
- Local vs. Global Shape: DCNNs can detect local shape features, such as edge segments and relations, but lack the ability to integrate these into a coherent global shape, which humans excel at.
- Texture vs. Shape: Unlike humans, DCNNs often prioritize surface texture as much as shape, showing that texture plays a significant role in their object recognition process.
- DCNN Limitations: When global shape is disrupted, DCNNs perform similarly as if the shape were intact, whereas humans struggle. Conversely, disrupting local contours significantly impairs DCNNs but not human recognition.

-> neural networks tend to rely more on textural
cues and local features than on global features, such as object shape


### [The contrasting roles of shape in human vision and convolutional neural networks](https://cpb-eu-w2.wpmucdn.com/blogs.bristol.ac.uk/dist/1/411/files/2020/07/cogsci19_Gaurav-paper.pdf)

TAG_CNN, TAG_SIM_TO_REAL

This paper discusses the difference between human vision and convolutional neural networks (CNNs) in object recognition, emphasizing how CNNs lack a natural shape-bias, unlike humans who primarily rely on shape to recognize objects. The key findings are:

1. CNNs' Lack of Shape-Bias: The experiments show that CNNs, when trained on datasets with additional noise-like masks (non-shape features), often ignore object shape altogether. Instead, CNNs use any feature that optimizes their performance, even if it's just a predictive noise pattern or a single pixel in the image.

2. Dataset Bias Influence: The study suggests that popular datasets, like CIFAR-10 and ImageNet, may contain biases or non-shape cues that CNNs can exploit for categorization. This reliance on non-shape features can result in CNNs behaving differently from humans, such as being easily confused by adversarial examples (e.g., "fooling images") or being overly sensitive to color or noise.

3. Implications for CNN Performance: The lack of shape-bias in CNNs could explain some of their vulnerabilities, like being easily manipulated by adversarial attacks, which humans are not typically susceptible to. This raises questions about the nature of CNN feature selection and its divergence from human visual processing.

4. Future Work: The study aims to investigate the computational benefits of introducing a shape-bias in CNNs to align their performance more closely with human vision and to explore the broader implications of such changes for robustness and generalization.

In summary, the paper highlights a fundamental difference between human vision and CNNs, suggesting that CNNs' reliance on non-shape features might be a source of some of their unpredictable behaviors, and proposes further exploration into how shape-bias can enhance their performance.

-> However, models trained only on synthetic datasets often poorly generalise to real data and bridging the gap between synthetic and real data remains an open research problem



### [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)





---
# Instance Segmentation

---

### [MaskRCNN](https://arxiv.org/pdf/1703.06870)

TAG_INSTANCE_SEGMENTATION, TAG_MASKRCNN

1. **Mask R-CNN Framework**: Mask R-CNN is an extension of Faster R-CNN, adding a branch to predict an object mask alongside the bounding box. This multi-task approach allows efficient object detection and high-quality instance segmentation.
   
2. **RoIAlign Layer**: The introduction of RoIAlign significantly improves accuracy, particularly at higher IoU thresholds (AP75). It resolves the issue of misalignment present in earlier methods like RoIPool and RoIWarp.
   
3. **Class-Agnostic vs. Class-Specific Masks**: Mask R-CNN can predict class-agnostic masks (one mask per object) almost as effectively as class-specific masks, which highlights its decoupling of classification and segmentation tasks.
   
4. **State-of-the-Art Performance**: Mask R-CNN achieves top results on the COCO dataset across instance segmentation, bounding-box detection, and keypoint detection, outperforming previous state-of-the-art models like FCIS and MNC.



### [Segmenting Unknown 3D Objects from Real Depth Images using Mask R-CNN Trained on Synthetic Data](https://arxiv.org/abs/1809.05825)

TAG_INSTANCE_SEGMENTATION, TAG_NOVEL_DATA, TAG_BIN_PICKING, TAG_SYNTHETIC_DATA, TAG_MASKRCNN

-> More synthetic data could further improve the model!
-> Depth alone is sufficient for segmentation
-> From synthetic data to real data works




### [Unseen Object Instance Segmentation for Robotic Environments](https://ieeexplore.ieee.org/abstract/document/9382336)

TAG_INSTANCE_SEGMENTATION, TAG_NOVEL_DATA, TAG_BIN_PICKING, TAG_SYNTHETIC_DATA

FIXME



### [A survey on instance segmentation: state of the art](https://link.springer.com/article/10.1007/s13735-020-00195-x)

TAG_INSTANCE_SEGMENTATION, TAG_NOVEL_DATA, TAG_GENERALIZATION

1. **Definition and Evolution**: The paper defines instance segmentation as the technique that combines object detection and semantic segmentation, providing distinct labels for separate instances of objects within the same class.
2. **Survey Scope**: It provides a comprehensive overview of instance segmentation, covering its background, techniques, and evolution, along with significant issues and challenges in the field.
3. **Techniques and Datasets**: The paper discusses various techniques for instance segmentation, including their strengths and weaknesses, and reviews popular datasets utilized in research.
4. **Future Research Directions**: The survey identifies major issues and challenges that present opportunities for future research, emphasizing the continuing evolution of instance segmentation with advancements in computing power.

Summary:
This survey paper offers an extensive overview of instance segmentation, highlighting its evolution from basic object detection to advanced pixel-level labeling of distinct object instances. It discusses key techniques and challenges, categorizing them and evaluating their strengths and weaknesses. Additionally, it reviews popular datasets used in the field and outlines future research opportunities to enhance instance segmentation techniques further.


### [A Survey on Object Instance Segmentation](https://link.springer.com/article/10.1007/s42979-022-01407-3)

TAG_INSTANCE_SEGMENTATION, TAG_DATA_AUGMENTATION, TAG_DEPTH_DATA

1. **Comprehensive Review of Techniques**: The paper surveys over 40 instance segmentation research papers, focusing on approaches based on deep learning, reinforcement learning, and transformers from 2016 to the present.
2. **Emergence of Transformers**: Recent advancements with transformers are highlighted for their superior performance in terms of accuracy and speed compared to previous methods, marking a notable trend in instance segmentation.
3. **Benchmarking Datasets**: The paper discusses commonly used datasets, providing a valuable resource for evaluating and comparing instance segmentation models.
4. **Challenges and Future Directions**: Challenges such as scalability, computational efficiency, and model accuracy are addressed, along with future opportunities for advancing instance segmentation research.

Summary
This survey paper provides an in-depth overview of instance segmentation, examining the evolution of deep learning, reinforcement learning, and transformer-based methods in this field. It presents a comparative analysis of techniques, highlights performance metrics, and reviews widely used datasets. The authors also discuss existing challenges and outline future research directions, offering a valuable reference for further development in instance segmentation.





---
# Bias

---

### [Can Biases in ImageNet Models Explain Generalization?](https://openaccess.thecvf.com/content/CVPR2024/papers/Gavrikov_Can_Biases_in_ImageNet_Models_Explain_Generalization_CVPR_2024_paper.pdf)

TAG_BIAS, TAG_GENERALIZATION

This paper examines the relationship between various biases—such as shape bias, spectral bias, and critical band bias—and the generalization capabilities of neural networks, specifically in image classification. Generalization refers to how well models perform on unseen data, both from within the distribution (rare, long-tail samples) and out-of-distribution (OOD) samples. The study critiques previous works that found correlations between these biases and generalization, proposing that while these biases might correlate with certain benchmarks or training modes, they fail to provide a holistic explanation for generalization across a broad range of conditions.

The researchers conducted a large-scale study using 48 ImageNet-trained models, all with the same architecture (ResNet-50) but trained with different methods. The results showed that biases like shape and spectral biases were insufficient to fully predict generalization performance. For example, adversarially-trained models exhibited stronger correlations with certain biases (like shape or low-frequency biases), but these findings did not generalize to the wider range of models.

Interestingly, some counterintuitive trends were observed, such as models with a strong texture bias performing better on in-distribution data, and models with a high-frequency bias improving generalization (though not adversarial robustness), despite high-frequency features being less discriminative. This highlights that no single bias can explain generalization in deep neural networks, but some biases, like a combination of low-bandwidth and high-frequency biases, might be necessary for achieving better generalization.

The paper concludes that generalization is too complex to be explained by any one bias alone. Future studies are needed to explore combinations of biases and other factors that could lead to better generalization, as well as improved benchmarks to better assess these relationships.

-> One Bias should not "destroys" the ability of generalizing (in case of ImageNet-trained and classification). So there is no need for redesigning the traindata, so that the bias is maybe reduced.



### [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231)

TAG_BIAS

This paper examines how Convolutional Neural Networks (CNNs) recognize objects, traditionally thought to rely on object shapes, but recent studies suggest a stronger reliance on textures. The authors tested this by evaluating CNNs and humans on images where texture and shape cues conflicted. They found that CNNs, especially those trained on ImageNet, are biased towards recognizing textures over shapes, which differs from human behavior. However, by training the same architecture (ResNet-50) on a stylized version of ImageNet that emphasizes shape over texture, the model shifted to a shape-based recognition strategy, aligning better with human visual performance. This shift also improved object detection and robustness to image distortions. The study highlights the importance of shape-based representations in neural networks for more human-like visual recognition and offers insights into CNN biases and their implications for future research.

-> CNN's are biased towards known Texture and stylizing the traindata improves/shifts this bias (in case of Classification with ImageNet-trained)
-> In ImageNet classification task, CNNs use local texture as the preliminary cue, while humans extract global shape information
-> Shape-Bias leads to robust CNN's



### [Towards Synthetic Data: Dealing with the Texture-Bias in Sim2real Learning](https://link.springer.com/chapter/10.1007/978-3-031-22216-0_42)

TAG_BIAS, TAG_SYNTHETIC_DATA, TAG_SIM_TO_REAL

- There is a Texture Bias in Semantic Segmentation ([DeepLabv3+](https://arxiv.org/pdf/1802.02611))
- Shape still keeps a fundamental role in decision making

-> we tested weather it is possible to bridge the gap in sim2real learning by using highly realistic textures



### [The Origins and Prevalence of Texture Bias in Convolutional Neural Networks](https://arxiv.org/pdf/1911.09071)

TAG_BIAS

The paper investigates the texture bias observed in ImageNet-trained Convolutional Neural Networks (CNNs), revealing that while CNNs typically classify images based on texture rather than shape, they can be trained to recognize shapes effectively. Key findings include:

- Influence of Training Objectives: Different unsupervised training methods and architectures have minimal but significant impacts on texture bias levels.
- Role of Data Augmentation: Employing less aggressive random cropping and adding naturalistic augmentations (like color distortion, noise, and blur) significantly reduces texture bias, enabling models to classify ambiguous images by shape more accurately.
- Outperformance on Test Sets: Models that favor shape classification surpass baseline performance on out-of-distribution test sets.
- Comparison to Human Vision: The differences between CNNs and human vision are attributed more to the training data statistics than to the model architectures or objectives.

The findings emphasize the need to adjust data statistics through augmentation to align CNNs' processing closer to human vision while acknowledging potential pitfalls in over-constraining models to mimic human judgments, which may include inherent biases.

This research contributes to understanding the divergences between human and machine vision and suggests methods for developing more robust and interpretable computer vision systems.

-> However, it is also possible to overcome the texture-bias by choosing more natural augmentation techniques. This paper suggest that CNNs are not inherently texture-biased and that the existing bias is the consequence of the unnatural augmentation techniques, such as random crop. Not only the augmentation of the training data but also changes in the model architecture can lead to the dominance of the shape-bias in the models



### [Shape-biased CNNs are Not Always Superior in Out-of-Distribution Robustness](https://openaccess.thecvf.com/content/WACV2024/papers/Qiu_Shape-Biased_CNNs_Are_Not_Always_Superior_in_Out-of-Distribution_Robustness_WACV_2024_paper.pdf)

TAG_BIAS, TAG_OOD

- The paper investigates the relationship between shape and texture information in achieving Out-of-Distribution (OOD) robustness in Deep Learning, particularly in shape-biased Convolutional Neural Networks (CNNs). While it's believed that shape-biased models would generally perform better due to their alignment with human cognitive decision-making, the authors find that shape information isn't always superior in distinguishing categories like animals and that shape-biased models don't consistently outperform others in various OOD scenarios, such as elastic transformations and blurs.
- To address these findings, the authors propose a novel method called Shape-Texture Adaptive Recombination (STAR). This method employs a category-balanced dataset to pretrain a debiased backbone and three specialized heads for extracting shape, texture, and debiased features. An instance-adaptive recombination head is then trained to adjust the contributions of these features dynamically for each instance.
- The experiments demonstrate that STAR achieves state-of-the-art OOD robustness across several scenarios, including image corruption, adversarial attacks, style shifts, and dataset shifts, validating the effectiveness of the proposed approach in enhancing robustness against diverse challenges.
- Overall, the research underscores the complexity of leveraging shape and texture information for improving model performance and suggests that a more nuanced approach, like STAR, can significantly enhance OOD robustness in CNNs.

-> Improving Method to create a dataset, which leads to a stable debiased CNN base method
-> Shape-Bias is not always wanted / the best way for good generalization (unknown data)
-> shape-aware models are not necessarily more accurate than their texture-biased counterparts



### [Increasing Shape Bias in ImageNet-Trained Networks Using Transfer Learning and Domain-Adversarial Methods](https://arxiv.org/abs/1907.12892)

TAG_BIAS, TAG_GENERALIZATION

- Texture and Color Bias in CNNs: CNNs have been found to exhibit a bias toward texture and color in their representations, which contrasts with human biological learning that emphasizes shape recognition.
- Enhancing Shape Bias: The study applies style-transfer techniques to remove texture clues and increase shape bias in CNNs, alongside using domain-adversarial training to further enhance this shape bias.
- Robustness Without Accuracy Gain: While the proposed methods improve the robustness and shape bias of the models across multiple datasets, they do not yield a clear increase in accuracy on the original datasets.
- Future Research Directions: Suggestions for future work include training models from scratch with domain-adversarial methods, experimenting with larger architectures, and considering scenarios where texture information might be beneficial for specific problems. The techniques could also be used as a data augmentation strategy during training.

-> shape-aware models are not necessarily more accurate than their texture-biased counterparts


### [Deep convolutional networks do not classify based on global object shape](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006613)

TAG_BIAS

-> Removing Texture Bias

FIXME



### [Informative Dropout for Robust Representation Learning: A Shape-bias Perspective](https://arxiv.org/pdf/2008.04254)

TAG_BIAS

-> Removing Texture Bias

FIXME



### [Learning Robust Shape-Based Features for Domain Generalization](https://ieeexplore.ieee.org/abstract/document/9050708)

TAG_BIAS

-> Removing Texture Bias

FIXME



### [Learning Visual Representations for Transfer Learning by Suppressing Texture](https://arxiv.org/abs/2011.01901)

TAG_BIAS

- Texture Over-Emphasis in CNNs: CNNs, especially in self-supervised learning, tend to rely heavily on texture information, which can act as a shortcut and prevent the network from learning higher-level representations.
- Anisotropic Diffusion for Texture Suppression: The authors propose using anisotropic diffusion to suppress texture and retain more edge information during training, helping CNNs focus on higher-level cues rather than low-level texture features.
- Improved Performance: This method shows significant improvements in both supervised and self-supervised learning tasks across five diverse datasets, with particularly strong results in transfer learning, achieving up to 11.49% improvement on Sketch-ImageNet and Synthetic-DTD datasets.
- Broad Applicability: The approach is simple to implement and can be integrated into various computer vision tasks, enhancing model generalization and transfer learning performance.

-> Removing Texture Bias with Style Transfer



### [Towards Shape Biased Unsupervised Representation Learning for Domain Generalization](https://arxiv.org/pdf/1909.08245)

TAG_BIAS

-> Shape-Bias leads to robust CNN's

FIXME



### [The shape and simplicity biases of adversarially robust ImageNet-trained CNNs](https://arxiv.org/pdf/2006.09373)

TAG_BIAS

-> Shape-Bias leads to robust CNN's

FIXME



### [Delving Deep into the Generalization of Vision Transformers under Distribution Shifts](https://arxiv.org/pdf/2106.07617)

TAG_BIAS

-> Shape-Bias leads to robust CNN's

FIXME



### [Informative Dropout for Robust Representation Learning: A Shape-bias Perspective](https://arxiv.org/pdf/2008.04254)

TAG_BIAS

-> Shape-Bias leads to robust CNN's

FIXME



### [Learning Robust Shape-Based Features for Domain Generalization](https://ieeexplore.ieee.org/abstract/document/9050708)

TAG_BIAS

-> Shape-Bias leads to robust CNN's

FIXME



### [Reducing Texture Bias of Deep Neural Networks via Edge Enhancing Diffusion](https://arxiv.org/pdf/2402.09530)

TAG_BIAS

FIXME



### [SHAPE-TEXTURE DEBIASED NEURAL NETWORK TRAINING](https://www.researchgate.net/profile/Cihang-Xie/publication/344639530_Shape-Texture_Debiased_Neural_Network_Training/links/5f886a5b92851c14bccbc93d/Shape-Texture-Debiased-Neural-Network-Training.pdf)

TAG_BIAS

1. **Shape-Texture Bias**: CNNs tend to bias towards either shape or texture, depending on the training dataset, leading to degraded performance.
2. **Debiasing Method**: The authors propose a shape-texture debiased learning algorithm that uses images with conflicting shape and texture cues and provides supervision for both aspects simultaneously.
3. **Performance Improvements**: The method enhances CNN performance across several benchmarks, including ImageNet, adversarial robustness, and data augmentation strategies like Mixup and CutMix.
4. **Complementary Cues**: Shape and texture are complementary in image recognition tasks, and balanced learning from both improves accuracy and robustness.

Summary:
This paper introduces a shape-texture debiased learning approach to address the bias in CNNs towards either shape or texture. The method augments training datasets with conflicting shape-texture images and applies supervision to both cues simultaneously. This debiasing strategy leads to substantial improvements in accuracy and robustness on various image recognition benchmarks and adversarial defenses, showing that shape and texture are complementary in neural networks.




### [UNIVERSAL ADVERSARIAL ROBUSTNESS OF TEXTURE AND SHAPE-BIASED MODELS](https://arxiv.org/abs/1911.10364)

TAG_BIAS

- Shape-Bias and Adversarial Robustness: The study investigates the impact of increasing shape-bias in deep neural networks (DNNs) on their robustness to Universal Adversarial Perturbations (UAPs), finding that shape-biased models do not significantly enhance adversarial robustness compared to texture-biased models.
- Vulnerability to Attacks: Shape-biased models exhibit similar vulnerabilities to universal adversarial attacks as texture-biased models, indicating that simply emphasizing shape does not inherently protect against these types of adversarial perturbations.
- Using UAPs for Evaluation: The research demonstrates how UAPs can be employed as a tool to evaluate and compare the robustness of models trained with varying degrees of texture and shape bias.
- Ensemble Approach for Improvement: The authors propose using ensembles of both texture and shape-biased models, which can enhance universal adversarial robustness while maintaining strong performance on clean data, suggesting a potential strategy for improving model resilience.


### [Shape or Texture: Understanding Discriminative Features in CNNs](https://arxiv.org/abs/2101.11604)

TAG_BIAS, TAG_CNN

1. **Texture Bias in CNNs**: CNNs exhibit a bias toward texture when predicting object categories, often prioritizing texture cues over shape, even in stylized images.
2. **Shape Information Encoding**: CNNs encode significant shape information across the network, but correct shape-based object categorization mostly occurs in the later layers.
3. **Effect of Removing Texture**: Removing texture from images significantly harms the network’s ability to associate shape with the correct category.
4. **Neuron-Specific Dependency**: CNNs rely differently on shape- or texture-specific neurons, affecting their performance based on the neurons’ presence or absence.

Summary:
This paper systematically examines how Convolutional Neural Networks (CNNs) encode shape information across their layers. Although CNNs are biased towards texture, they also retain significant shape information throughout the network. Proper categorization based on shape occurs mainly in the final layers. Experiments reveal that removing texture information hurts performance, and targeting specific neurons affects the model's reliance on shape or texture. These insights contribute to a better understanding of CNN behavior, guiding the design of more robust and reliable computer vision models.



### [Trapped in Texture Bias? A Large Scale Comparison of Deep Instance Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-20074-8_35)

TAG_INSTANCE_SEGMENTATION, TAG_BIN_PICKING, TAG_MASKRCNN, TAG_SYNTHETIC_DATA, TAG_DEPTH_DATA, TAG_BIAS, TAG_NOVEL_DATA, TAG_GENERALIZATION, TAG_OOD, TAG_SIM_TO_REAL, TAG_CNN

1. **Texture Bias in Instance Segmentation**: Most deep learning models for instance segmentation exhibit a noticeable bias toward texture, impacting their robustness when faced with out-of-distribution textures.
2. **Top Robust Models**: YOLACT++, SOTR, and SOLOv2 demonstrated significantly better robustness to out-of-distribution texture compared to other frameworks like Cascade and Mask R-CNN.
3. **Architecture Over Training**: Deeper and dynamic architectures improve robustness more than training schedules, data augmentation, or pre-training techniques.
4. **Comprehensive Evaluation**: The study conducted 4,148 evaluations across 68 models and 61 MS COCO dataset versions, providing a solid baseline for texture robustness in instance segmentation models.

Summary:
This paper evaluates the robustness of deep learning models for instance segmentation, focusing on their ability to generalize to novel, out-of-distribution textures. By testing various architectures such as YOLACT++, SOTR, and SOLOv2, the study finds that deeper and dynamic models significantly improve robustness to texture variations. The research provides a comprehensive analysis of texture bias across instance segmentation methods and aims to guide future design choices in the field.


### [Exploring Shape and Texture Bias in Object Detection](http://viplab.snu.ac.kr/viplab/courses/mlvu_2024_1/projects/02.pdf)

TAG_BIAS

1. **Low Texture Bias in Object Detection**: Unlike image classification models, object detection models exhibit a higher shape bias and lower texture bias due to the task's focus on object localization and bounding box alignment with object silhouettes.
2. **Improved Augmentation for Object Detection**: The paper presents a novel augmentation method specifically designed for object detection, improving upon previous texture bias evaluation methods used in classification tasks.
3. **Influence of Loss Function and Task**: The shape bias in object detection models is driven by the bounding box loss, which emphasizes aligning with object shapes, highlighting the importance of task-specific loss functions in mitigating texture bias.
4. **Recommendations for Image Classification**: The authors propose introducing localization techniques and modifying datasets to enhance shape learning in image classification tasks, which are typically more texture-biased.

Summary:
This paper explores the texture and shape bias in object detection models, finding that these models have a higher shape bias compared to image classification models, thanks to the bounding box loss that promotes shape alignment. The authors introduce new augmentation methods suited for object detection and recommend improvements for reducing texture bias in classification tasks, such as localization techniques and shape-focused training.



### [A Competition of Shape and Texture Bias by Multi-view Image Representation](https://link.springer.com/chapter/10.1007/978-3-030-88013-2_12)

TAG_BIAS

1. **CNNs Lack Fixed Bias**: Contrary to popular belief, CNNs do not exhibit a fixed shape or texture bias. Their biases adapt based on the internal properties of the training data.
2. **Lazy Learning**: CNNs prefer to learn low-level features like texture, and only move on to higher-level features like shape when low-level cues are insufficient for task performance.
3. **Multi-View Image Representation**: The study segments image features into shape, texture, and background components, using a combination of losses (reconstruction, discrepancy, and classification) to learn more robust representations.
4. **Implications for Explainability**: The research sheds light on the interpretability of CNN models, suggesting that understanding their adaptive feature learning could help design more robust models.

Summary:
This paper challenges the notion of fixed texture or shape bias in CNNs, showing that these biases change depending on the training data. CNNs learn features in a "lazy" manner, focusing on simpler features like texture before learning more complex ones like shape. By using a multi-view representation that separates shape, texture, and background, the authors propose a more interpretable model that adapts based on task demands. This has important implications for improving CNN robustness and understanding model behavior.



### [On the Texture Bias for Few-Shot CNN Segmentation](https://arxiv.org/abs/2003.04052)

TAG_BIAS

1. **Texture Bias Mitigation**: The paper presents a novel architecture aimed at reducing texture bias in Convolutional Neural Networks (CNNs) for few-shot semantic segmentation, proposing the integration of a pyramid of Difference of Gaussians (DoG) to diminish high-frequency local components in the feature space.
2. **Bi-Directional LSTM Integration**: To effectively merge the multi-scale feature maps generated by the DoG approach, the architecture employs a bi-directional convolutional long-short-term-memory (LSTM) model, facilitating better representation of spatial features.
3. **Benchmark Performance**: Extensive experiments on established few-shot segmentation benchmarks (Pascal i5, COCO-20i, and FSS-1000) demonstrate that the proposed model outperforms state-of-the-art methods on two datasets while maintaining a lightweight architecture.
4. **State-of-the-Art Results**: The model achieves new state-of-the-art performance on several few-shot semantic segmentation tasks, showcasing its effectiveness in scenarios with limited labeled data.

### Summary:
This study introduces a novel segmentation network that addresses the challenge of few-shot learning by reducing the texture bias inherent in CNNs. By integrating a pyramid of Difference of Gaussians to attenuate high-frequency features and employing bi-directional convolutional LSTMs to merge multi-scale representations, the proposed model surpasses existing state-of-the-art approaches in several benchmarks. This research contributes significantly to the field of semantic segmentation, particularly in low-labeled data scenarios.



### [Shape Prior is Not All You Need: Discovering Balance Between Texture and Shape Bias in CNN](https://www.researchgate.net/publication/368728987_Shape_Prior_is_Not_All_You_Need_Discovering_Balance_Between_Texture_and_Shape_Bias_in_CNN)

TAG_BIAS, TAG_GENERALIZATION, TAG_CNN

1. **Texture vs. Shape Bias**: The paper highlights that Convolutional Neural Networks (CNNs), even after fine-tuning, exhibit a bias towards texture over shape, which is influenced by the distribution of the downstream datasets.
2. **AdaBA Method**: The authors propose a novel method called AdaBA for quantitatively analyzing and illustrating the bias landscape of CNNs, addressing limitations of previous methods and focusing on fine-tuned models.
3. **Granular Labeling Scheme**: To mitigate texture bias, the study introduces a granular labeling scheme that redesigns the label space, aiming to achieve a balance between texture and shape biases, which has been empirically shown to enhance classification and out-of-distribution (OOD) detection performance.
4. **Performance Improvement**: The granular labeling scheme not only balances the biases but also improves the representation power of CNNs, leading to better performance in tasks like classification and OOD detection.

Summary:
This paper investigates the texture and shape biases of Convolutional Neural Networks (CNNs), especially those fine-tuned beyond their initial training on ImageNet. By introducing the AdaBA method for analyzing these biases and proposing a granular labeling scheme to balance the biases, the authors demonstrate significant improvements in CNN performance on classification tasks and out-of-distribution detection. The findings aim to enhance the understanding of CNN behavior and provide effective solutions to reduce texture bias.



### [Shape-aware Instance Segmentation](https://www.researchgate.net/publication/311573635_Shape-aware_Instance_Segmentation)

TAG_BIAS, TAG_INSTANCE_SEGMENTATION

-> New Model for Instance Segmentation with Shape Awareness
-> Want to fix the problem, that when the object bounding predictions are bad than the masks are also bad



### [ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors](https://www.researchgate.net/publication/332300208_ShapeMask_Learning_to_Segment_Novel_Objects_by_Refining_Shape_Priors)

TAG_BIAS, TAG_INSTANCE_SEGMENTATION

-> New Model for Instance Segmentation with Shape Awareness and for novel objects



### [On the Influence of Shape, Texture and Color for Learning Semantic Segmentation](https://arxiv.org/html/2410.14878v1)

TAG_BIAS, TAG_CNN, TAG_GENERALIZATION

1. **Cue-Specific Learning in DNNs**: The study examines how deep neural networks (DNNs) for semantic segmentation learn from different image cues—shape, texture, and color—and their combined influence on learning success.
2. **Synergy of Shape and Color**: A combination of shape and color (without texture) achieves strong segmentation performance, suggesting that these cues are crucial when working together.
3. **Cue Importance by Context**: Shape cues are particularly significant near semantic boundaries, while texture is important for classes covering large image regions.
4. **Generalizable Framework**: The paper introduces a generic procedure to create cue-specific datasets, which can be applied to both CNNs and transformers to study cue-specific learning across datasets.

Summary:
This paper explores how DNNs for semantic segmentation learn from individual and combined image cues—shape, texture, and color—by creating cue-specific datasets from real-world and synthetic data. The study reveals that while no single cue dominates, the combination of shape and color is particularly effective, especially near boundaries. The proposed cue extraction procedure provides a framework for further research on bias in pre-trained models, applicable to various neural network architectures like CNNs and transformers.





---
# Generalization

---

### [Texture Learning Domain Randomization for Domain Generalized Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Kim_Texture_Learning_Domain_Randomization_for_Domain_Generalized_Segmentation_ICCV_2023_paper.html)

TAG_GENERALIZATION

FIXME





---
# RGB-D

---

### [Bin-picking of novel objects through category-agnostic-segmentation: RGB matters](https://arxiv.org/abs/2312.16741)

TAG_INSTANCE_SEGMENTATION, TAG_BIN_PICKING, TAG_SIM_TO_REAL

1. **Category-Agnostic Instance Segmentation**: The paper focuses on developing an instance segmentation approach for robotic manipulation that works independently of object categories, making it versatile for applications like bin-picking in dynamic environments.

2. **Simulation-Based Training for Real-World Transfer**: The method relies on simulation-based training, utilizing domain randomization to ensure successful transfer to real-world scenarios without needing real-world training samples.

3. **Handling Transparent and Semi-Transparent Objects**: Unlike many depth-based methods, the proposed approach is effective at handling transparent and semi-transparent objects, historically difficult for robotic grasping tasks.

4. **High Performance in Real-World Bin-Picking**: The bin-picking framework achieves 98% accuracy for opaque objects and 97% for non-opaque objects, outperforming state-of-the-art baselines in real-world tests.

Summary:
This paper introduces a novel approach to category-agnostic instance segmentation for robotic manipulation, particularly suited for bin-picking applications. The method addresses common issues such as noisy depth sensors and the challenge of segmenting transparent objects. By leveraging simulation-based training and domain randomization, the solution achieves impressive performance in real-world scenarios, even without real-world training data. It demonstrates high accuracy in bin-picking tasks and provides a framework that significantly enhances efficiency in warehouse applications.

-> Achieves outstanding performance with only RGB Images (Also with non-opaque objects and can also perfrom very well without a training on novel objects)
-> Say not much about their technology and their data



### [Benchmarking of deep learning algorithms for 3D instance segmentation of confocal image datasets](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009879)

TAG_DEPTH_DATA, TAG_MASKRCNN, TAG_SIM_TO_REAL, TAG_INSTANCE_SEGMENTATION, TAG_DATA_AUGMENTATION

1. **Benchmarking Process:** The study evaluates five segmentation pipelines (four DL and one non-DL) on 3D microscopy images, using a common dataset and uniform evaluation metrics. This ensures a consistent and comparative benchmarking framework. 
2. **Performance Variations:** End-to-end 3D pipelines, such as PlantSeg and UNet+WS, perform more accurately than adapted 2D pipelines (e.g., Cellpose). Differences in architecture and processing approaches (e.g., graph partitioning vs. watershed) contribute to varying segmentation accuracies, especially on complex 3D structures.
3. **Robustness to Artifacts:** Image artifacts like blur, exposure variations, and Gaussian noise impact segmentation accuracies. However, some DL models (e.g., PlantSeg) adapt better through training with augmented data, highlighting the importance of data augmentation in enhancing robustness.
4. **Interactive Quality Evaluation:** A MorphoNet-based visualization approach was developed to assess segmentation quality in 3D, providing an intuitive interface to inspect segmentation errors at different cell depths and enhancing interpretability.

Summary
This paper systematically benchmarks 3D microscopy segmentation pipelines, comparing the performances of DL and non-DL approaches on uniform data and evaluation criteria. The study identifies performance variability across pipelines due to their unique architectures and response to image artifacts, underscoring the need for tailored evaluation metrics and data augmentations. The proposed MorphoNet-based interactive visualization tool aids in assessing segmentation quality in 3D, offering insights into segmentation accuracy at a finer, cell-layered level.

-> depth data is important for segmentation complex 3D structures
-> data augmentation is important (especially Blur and Exposure)



### [Exploiting Depth Information for Wildlife Monitoring](https://arxiv.org/abs/2102.05607)

TAG_DEPTH_DATA

1. **D-Mask R-CNN Development**: The paper introduces D-Mask R-CNN, a novel method for instance segmentation that combines standard detection with depth estimation to improve the identification and delineation of individual animals in images and video clips.
2. **Performance Improvement**: Experimental evaluations show that D-Mask R-CNN achieves significantly better average precision (AP) scores compared to traditional Mask R-CNN, with improvements of 9.81% and 8.93% for bounding boxes and segmentation masks, respectively, on a synthetic dataset.
3. **Real-World Application**: The approach was tested in a zoo setting using an RGB-D camera trap, achieving AP scores of 59.94% and 37.27% for deer detection, demonstrating its practical applicability in biodiversity research.
4. **Future Work Directions**: Future plans include expanding the RGB-D camera trap dataset, deploying stereo-based RGB-D camera traps for enhanced depth estimation, and applying D-Mask R-CNN to automate ecological models and incorporate additional tasks like keypoint detection.

Summary:
This study presents an innovative approach to wildlife detection using camera traps enhanced with depth estimation. The newly developed D-Mask R-CNN method significantly outperforms traditional image-only detection techniques, as evidenced by higher average precision scores in both synthetic and real-world scenarios. The research highlights the potential of combining depth data with instance segmentation to facilitate more efficient biodiversity monitoring and lays the groundwork for future advancements in automated ecological studies.



### [Deep Learning Based 3D Segmentation: A Survey](https://arxiv.org/abs/2103.05423)

TAG_DEPTH_DATA

1. **Comprehensive Survey on 3D Segmentation**: The paper provides an in-depth survey of over 220 works related to 3D segmentation using deep learning techniques, including 3D semantic, instance, and part segmentation.
2. **Comparison of State-of-the-Art Methods**: The survey evaluates the strengths and weaknesses of various methodologies, highlighting the current state-of-the-art in 3D segmentation across different datasets and applications.
3. **Emerging Research Directions**: The paper identifies promising future research directions, such as developing novel architectures specifically for 3D segmentation, integrating multi-modal data, and improving techniques for large-scale and dynamic environments.
4. **Applications in Various Domains**: 3D segmentation is noted to have significant applications in fields like autonomous driving, robotics, medical imaging, and virtual reality.

Summary:
This paper provides a thorough survey of the latest advancements in 3D segmentation, covering over 220 methodologies in the last six years. It evaluates state-of-the-art deep learning techniques used in 3D semantic, instance, and part segmentation, highlighting their performance on benchmark datasets. The paper also discusses future research opportunities, including novel architectures, multi-modal data integration, and handling large-scale 3D environments, emphasizing the importance of 3D segmentation across various domains.


### [Depth-aware object instance segmentation](https://www.semanticscholar.org/paper/Depth-aware-object-instance-segmentation-Ye-Liu/64b9675e924974fdec78a7272b27c7e7ec63a608)

TAG_INSTANCE_SEGMENTATION, TAG_DEPTH_DATA, TAG_CNN

1. **Depth-Aware Object Instance Segmentation**: The paper introduces a novel approach to object instance segmentation that incorporates depth information to resolve overlapping instances.
2. **Three-Step Approach**: The method consists of object instance detection, category-specific instance segmentation, and depth-aware ordering to improve pixel labeling in complex scenes.
3. **Ambiguity Resolution**: Depth information helps reduce label ambiguity, particularly in cases where object instances overlap in the image.
4. **Performance**: The approach demonstrates competitive performance on the PASCAL VOC 2012 benchmark compared to state-of-the-art methods.

Summary:
This paper presents a depth-aware approach for object instance segmentation, addressing the challenge of overlapping instances by utilizing depth information. The method follows a three-step process involving object detection, category-specific segmentation, and depth-aware ordering, which effectively reduces ambiguity in pixel labeling. Experimental results show strong performance on the PASCAL VOC 2012 benchmark.



### [Depth-aware CNN for RGB-D Segmentation](https://arxiv.org/abs/1803.06791)

TAG_DEPTH_DATA,TAG_INSTANCE_SEGMENTATION, TAG_CNN

1. **Depth-aware CNN**: The paper introduces two new operations—**depth-aware convolution** and **depth-aware average pooling**—to seamlessly incorporate geometric information into CNNs using depth similarity between pixels.
2. **Efficiency and Flexibility**: These depth-aware operations enhance CNNs without adding extra parameters or computational complexity, making them efficient and easily integrated into existing models.
3. **Performance Boost**: The proposed method significantly improves RGB-D segmentation performance over standard CNNs, as demonstrated through extensive experiments and comparisons with state-of-the-art methods.
4. **Future Work**: The authors suggest extending depth-aware CNNs to tasks like 3D detection and instance segmentation, as well as applying them to more complex datasets and using other 3D data formats such as LiDAR point clouds.

Summary:
This paper presents a novel depth-aware CNN framework that enhances standard CNNs by incorporating geometric information through depth-aware convolution and pooling operations. These operations leverage depth similarity, boosting performance in RGB-D segmentation without increasing computational complexity. The depth-aware CNN framework is flexible, efficient, and shows promise for future applications in 3D detection and instance segmentation tasks.



### [RGB Matters: Learning 7-DoF Grasp Poses on Monocular RGBD Images](https://arxiv.org/abs/2103.02184)

TAG_DEPTH_DATA, TAG_INSTANCE_SEGMENTATION

FIXME



### [Robot Unknown Objects Instance Segmentation Based on Collaborative Weight Assignment RGB–Depth Fusion Strategy](https://ieeexplore.ieee.org/abstract/document/10317822)

TAG_DEPTH_DATA, TAG_INSTANCE_SEGMENTATION

FIXME



### [Outdoor RGBD Instance Segmentation With Residual Regretting Learning](https://ieeexplore.ieee.org/abstract/document/9016374)

TAG_DEPTH_DATA, TAG_INSTANCE_SEGMENTATION

FIXME



### [ClusterNet: 3D Instance Segmentation in RGB-D Images](https://arxiv.org/abs/1807.08894)

TAG_DEPTH_DATA, TAG_INSTANCE_SEGMENTATION

FIXME



### [3D Instance Segmentation Using Deep Learning on RGB-D Indoor Data](https://arxiv.org/abs/2406.14581)

TAG_DEPTH_DATA, TAG_INSTANCE_SEGMENTATION

FIXME



### [Learning RGB-D Feature Embeddings for Unseen Object Instance Segmentation](https://proceedings.mlr.press/v155/xiang21a.html)

TAG_DEPTH_DATA, TAG_INSTANCE_SEGMENTATION, TAG_SYNTHETIC_DATA

1. **Unseen Object Instance Segmentation (UOIS)**: The paper introduces a new method for segmenting unseen objects in cluttered scenes using RGB-D feature embeddings learned from synthetic data.
2. **Metric Learning for Feature Embedding**: A metric learning loss function is applied to learn pixel-wise embeddings, ensuring that pixels from the same object are grouped closely and those from different objects are separated.
3. **Two-Stage Clustering Algorithm**: A novel two-stage clustering algorithm is proposed to improve segmentation accuracy, especially in scenes where objects are close together.
4. **Transferability from Synthetic to Real Data**: The method demonstrates that non-photorealistic synthetic RGB and depth images can be used to effectively learn feature embeddings that generalize to real-world images.

Summary:
This paper presents a method for unseen object instance segmentation (UOIS) using RGB-D feature embeddings learned from synthetic data. The approach leverages a metric learning loss to embed pixels from the same object closely in feature space, and introduces a two-stage clustering algorithm to segment objects in cluttered scenes. The method shows strong transferability from synthetic RGB-D data to real-world images, improving segmentation performance in previously unseen environments.



### [The Best of Both Modes: Separately Leveraging RGB and Depth for Unseen Object Instance Segmentation](http://proceedings.mlr.press/v100/xie20b.html)

TAG_DEPTH_DATA, TAG_INSTANCE_SEGMENTATION

FIXME




---
# Sim-to-Real

---


### [Sim2real transfer learning for 3D human pose estimation: motion to the rescue](https://arxiv.org/abs/1907.02499)

TAG_SIM_TO_REAL, TAG_SYNTHETIC_DATA

- Sim2Real Problem in 3D Human Pose Estimation: Models trained on synthetic data often generalize poorly to real-world data. This is particularly problematic for tasks like 3D human pose estimation, where obtaining real labeled 3D poses is challenging.
- Motion as a Key Bridging Factor: The study shows that while standard neural networks perform poorly when trained on synthetic RGB images, they perform significantly better when motion cues—such as optical flow and 2D keypoint motion—are used. This suggests that motion information can help bridge the sim2real gap.
- State-of-the-Art Performance on 3D Poses in the Wild: Despite being trained only on synthetic data from the SURREAL dataset, the model achieved results on the challenging 3D Poses in the Wild dataset comparable to state-of-the-art methods trained on real data.
- Broader Applications of Motion Cues: The findings suggest potential applications beyond human pose estimation, such as in robotics, where object motion and camera movement can similarly provide critical cues for estimating object pose. The authors also propose using self-supervised learning to refine RGB models after motion-based pose estimation.

-> However, models trained only on synthetic datasets often poorly generalise to real data and bridging the gap between synthetic and real data remains an open research problem




---
# Synthetic Data

---

### [Driving in the Matrix: Can Virtual Worlds Replace Human-Generated Annotations for Real World Tasks?](https://arxiv.org/abs/1610.01983)

TAG_SYNTHETIC_DATA

This paper explores a method of using photo-realistic computer-generated images from a simulation engine to create annotated datasets for training deep learning algorithms. The key points of the paper are:

1. **Problem with Annotated Data**: Deep learning advances, particularly in computer vision and robotics, heavily rely on large amounts of human-annotated data. The process of manual annotation is time-consuming and has become a bottleneck for further progress.

2. **Synthetic Data for Training**: The authors propose generating synthetic annotated data from a highly realistic visual simulator. This synthetic data is used to train deep learning models, bypassing the need for human annotation. The study shows that models trained exclusively on simulated data outperform those trained on real-world, human-annotated data when tested on the KITTI dataset for vehicle detection.

3. **Effectiveness on Real-World Data**: Despite being trained solely on synthetic data, the deep learning models perform well on real-world data, demonstrating the potential of synthetic data for training sensor-based classification systems, like those used in self-driving cars.

4. **Addressing Dataset Bias**: The study emphasizes how traditional training on modest real-world datasets can introduce biases that limit the effectiveness of models. Synthetic data can mitigate this by providing a larger variety of training examples and greater control over the features present in the images.

5. **Future Research Directions**: The authors suggest further research in areas like testing the effectiveness of simulated data across different deep learning architectures, enhancing network depth to capitalize on large synthetic datasets, and using active learning to maintain high performance with smaller subsets of simulation data. Additionally, they propose exploring more complex and specific simulations to further improve real-world performance.

In summary, this paper demonstrates that synthetic data from realistic simulators can accelerate deep learning's application to problems like vehicle detection, reducing reliance on manually annotated real-world data while achieving high performance on real-world tasks.



### [The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes](https://www.researchgate.net/publication/311610734_The_SYNTHIA_Dataset_A_Large_Collection_of_Synthetic_Images_for_Semantic_Segmentation_of_Urban_Scenes/link/5d921593458515202b754b44/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19)

TAG_SYNTHETIC_DATA

This paper focuses on improving vision-based semantic segmentation in urban scenarios, a critical component for autonomous driving systems. It emphasizes the utility of deep convolutional neural networks (DCNNs) for this task, which have shown great promise in recent advancements. However, DCNNs require extensive training data with pixel-level annotations, which are labor-intensive to produce, especially for semantic segmentation tasks.

To address this challenge, the authors propose using synthetic data generated from a virtual world, specifically through the creation of the **SYNTHIA** dataset. Key points of the paper include:

1. **SYNTHIA Dataset**: The paper introduces SYNTHIA, a large collection of synthetic urban driving scenes, containing over 213,400 images with pixel-level semantic annotations. The dataset includes a variety of conditions (e.g., different seasons, weather, illumination) and multiple viewpoints, providing rich and diverse data. The images are generated in a simulated city environment and come with pixel-level semantic annotations and depth information.

2. **Purpose and Use**: The primary aim of SYNTHIA is to address the need for large, annotated datasets for training DCNNs in semantic segmentation. By combining SYNTHIA with real-world images that have manual annotations, the authors examine the impact of synthetic data on the performance of DCNN models.

3. **Experimental Results**: Experiments conducted with DCNNs demonstrated that including SYNTHIA in the training process significantly improves performance on semantic segmentation tasks when evaluated on real-world datasets. The synthetic images alone were shown to perform well, but their real impact was seen when combined with real-world data, leading to a dramatic boost in segmentation accuracy.

4. **Contribution to Semantic Segmentation Research**: The paper suggests that SYNTHIA will be a valuable resource for researchers working on semantic segmentation, particularly in autonomous driving. The synthetic data generated from the virtual environment helps alleviate the need for costly and time-consuming manual annotations while still improving the performance of state-of-the-art models.

In summary, the SYNTHIA dataset is presented as a powerful tool to enhance semantic segmentation for autonomous driving, showing that synthetic data, when combined with real-world data, can significantly improve DCNN performance and boost research in this area.



### [Harnessing Synthetic Datasets: The Role of Shape Bias in Deep Neural Network Generalization](https://arxiv.org/abs/2311.06224)

TAG_BIAS, TAG_SYNTHETIC_DATA

1. **Shape Bias and Network Performance**: The study finds that shape bias varies across different neural network architectures and supervision types, questioning its reliability as a predictor for generalization and its alignment with human recognition capabilities.
2. **Unreliability of Shape Bias for Generalization**: It is highlighted that relying solely on shape bias to estimate a model's generalization performance is unreliable, as it is intertwined with the diversity and naturalism of the synthetic datasets.
3. **Shape Bias as a Diversity Metric**: The authors propose a novel interpretation of shape bias, suggesting it can serve as a proxy for estimating the diversity of samples within a synthetic dataset, rather than merely indicating generalization capabilities.
4. **Implications for Synthetic Data**: The paper aims to clarify the implications of using synthetic data in deep learning, addressing concerns about its quality and its influence on model generalization.

Summary:
This paper investigates the relationship between shape bias in neural networks trained on synthetic datasets and the implications for generalization to real-world samples. The authors highlight the variability of shape bias across different architectures and the entangled effects of diversity and naturalism on generalization. They propose using shape bias as a metric for assessing the diversity of synthetic samples instead of a direct predictor of model performance. The findings encourage further exploration of the properties of synthetic datasets and strategies for their effective design.






---
### Data Augmentation

---

### [Data Augmentation in Classification and Segmentation: A Survey and New Strategies](https://www.mdpi.com/2313-433X/9/2/46)

TAG_DATA_AUGMENTATION, TAG_INSTANCE_SEGMENTATION, TAG_DATA_AMOUNT

1. **Need for Data Augmentation**: The paper highlights that deep learning models, particularly convolutional neural networks (CNNs), require large datasets to avoid overfitting, which is often a challenge due to limited data availability in real-world scenarios.
2. **Survey of Techniques**: It surveys existing data augmentation techniques in computer vision, specifically for tasks like segmentation and classification, and proposes new strategies to enhance performance.
3. **Random Local Rotation (RLR)**: The authors introduce a novel data augmentation strategy called **random local rotation (RLR)**, which randomly selects circular regions in an image and rotates them at random angles, addressing the limitations of traditional rotation techniques.
4. **Experimental Results**: The RLR strategy consistently outperforms traditional data augmentation methods in both image classification and some segmentation tasks, demonstrating its effectiveness in improving model performance.

Summary
The paper discusses the critical role of data augmentation in overcoming the overfitting problem faced by deep learning models, particularly in scenarios with limited data. It reviews existing techniques and introduces a new method, random local rotation (RLR), which enhances traditional augmentation strategies by focusing on local image information without introducing non-original pixel values. The experimental results show that RLR can significantly improve model performance in various computer vision tasks.



### [Image- and Instance-Level Data Augmentation for Occluded Instance Segmentation](https://www.researchgate.net/publication/375086608_Image-_and_Instance-Level_Data_Augmentation_for_Occluded_Instance_Segmentation)

TAG_INSTANCE_SEGMENTATION, TAG_DATA_AUGMENTATION, TAG_NOVEL_DATA, TAG_DATA_AMOUNT

1. **Addressing Limited Data**: The paper tackles the challenge of limited data in instance segmentation by employing image-level data augmentation techniques, including pixel-level and spatial-level augmentations.
2. **Balanced Occlusion Aware Copy-Paste (BOACP)**: The authors introduce a novel method called **Balanced Occlusion Aware Copy-Paste (BOACP)**, which not only increases the number of instances in images but also ensures a balance of occluded instances at the image level.
3. **Hybrid Task Cascade (HTC) Model**: The study utilizes the **Hybrid Task Cascade (HTC)** model based on CBSwin-Base and CBFPN, which demonstrates strong performance in segmentation tasks.
4. **Challenge Success**: The proposed methods achieved first place in the initial phase of the **DeepSportRadar Instance Segmentation Challenge** at the ACM MM-Sports 2023 Workshop, showcasing the effectiveness of their approach.

Summary
This paper addresses key challenges in instance segmentation, specifically limited data and occlusion issues, by utilizing image-level data augmentation and introducing a new method, Balanced Occlusion Aware Copy-Paste (BOACP). The authors leverage a Hybrid Task Cascade model, demonstrating that their methods enhance the model's performance on occluded instances. Their approach was validated through successful performance in a competitive challenge, highlighting its practical applicability.



### [A Deep Learning Image Data Augmentation Method for Single Tumor Segmentation](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.782988/full)

TAG_DATA_AUGMENTATION, TAG_USEFUL

-> Proofs the improvement when using Data Augmentation



### [Data augmentation and machine learning techniques for control strategy development in bio-polymerization process](https://www.sciencedirect.com/science/article/pii/S266649842200028X)

TAG_DATA_AUGMENTATION, TAG_NOT_SO_USEFUL

1. **Challenge of Limited Data**: The paper highlights the issue of **insufficient experimental data** in biochemistry, particularly in organic chemistry, exacerbated by the COVID-19 pandemic, which negatively impacts modeling performance.
2. **Data Augmentation Techniques**: To address the small sample size problem, the study employs **variational auto-encoders (VAEs)** and **generative adversarial networks (GANs)** for data augmentation, improving the performance of regression models in predicting molecular weights.
3. **Model Performance**: Among the machine learning algorithms tested, the **random forest model** enhanced by GAN data augmentation achieved the best results, with an R² value of **0.94** for the training set and **0.74** for the test set, indicating strong predictive capabilities.
4. **Future Improvements**: The authors note that the current models have limitations due to a **small number of attributes** in the dataset, suggesting that incorporating a more diverse dataset with multiple attributes could lead to significantly better model performance in future studies.

Summary
This paper addresses the challenge of limited data in biochemistry by proposing a machine learning approach for the bio-polymerization process. Using data augmentation techniques such as VAEs and GANs, the authors demonstrate significant improvements in model performance, particularly with the random forest algorithm. The study emphasizes the potential for these models to optimize processes without extensive experimentation and suggests that further research with more diverse datasets could enhance predictive accuracy.



### [Data Augmentation Methods for Semantic Segmentation-based Mobile Robot Perception System](https://scindeks-clanci.ceon.rs/data/pdf/1451-4869/2022/1451-48692203291J.pdf)

TAG_DATA_AUGMENTATION, TAG_NOVEL_DATA, TAG_GENERALIZATION, TAG_USEFUL

1. **Evaluation of 17 Techniques**: The paper evaluates **17 different data augmentation techniques** to enhance the performance of a deep learning model used in a mobile robot perception system, specifically for semantic segmentation tasks.
2. **Dataset Expansion**: The authors train the ResNet18 model on a small dataset of **159 images**, which, after applying the augmentation techniques, is expanded to **2607 images**. This significant increase in data is crucial for improving model performance.
3. **Improvement in Performance**: The best combination of data augmentation strategies resulted in a **6.2 point increase** in mean Intersection over Union (mIoU), achieving an overall mIoU of **93.2**. The most effective group of techniques identified were **noise addition methods**, which alone contributed to a **3.9 mIoU** increase.
4. **Future Research Directions**: The paper suggests that future studies should explore the performance of various deep learning models trained on datasets augmented with a combination of methods applied to individual images.

Summary
This paper provides a thorough evaluation of various data augmentation techniques for improving the accuracy of deep learning models used in mobile robot perception systems. By applying 17 different methods, the authors significantly increase the training dataset size and enhance the model's generalization capabilities. The results indicate that the combination of these techniques notably improves performance metrics, emphasizing the importance of noise addition methods. Future research is encouraged to investigate the effects of these augmentation strategies on different deep learning architectures.



### [Data augmentation: A comprehensive survey of modern approaches](https://www.sciencedirect.com/science/article/pii/S2590005622000911)

TAG_DATA_AUGMENTATION, TAG_NOVEL_DATA, TAG_GENERALIZATION

1. **Importance of Data Quality**: The paper emphasizes that **quality annotated data** is crucial for the performance of machine learning models, but collecting and annotating this data is often resource-intensive and time-consuming.
2. **Role of Data Augmentation**: Data augmentation is highlighted as the most effective strategy for increasing the **volume, quality, and diversity** of training data, particularly in scenarios where obtaining sufficient data is not feasible.
3. **Comprehensive Survey**: The paper presents an extensive survey of various **data augmentation techniques** in computer vision, covering both traditional methods and modern approaches, including **deep learning strategies, feature-level augmentation, and data synthesis methods** like generative adversarial networks and neural rendering.
4. **Challenges in Implementation**: Despite the benefits of data augmentation, the paper discusses significant challenges, such as the difficulty in predicting which transformations will yield the best results and the computational costs associated with automated augmentation techniques.

Summary
This paper provides a comprehensive review of data augmentation methods applicable to computer vision, addressing the critical need for high-quality annotated data in machine learning. It highlights the effectiveness of various augmentation strategies, including both traditional and modern techniques, while discussing the challenges developers face in selecting the right methods for their specific tasks. The authors stress the importance of ongoing research in data augmentation to enhance model performance, especially in data-limited scenarios.



### [Performance improvement of Deep Learning Models using image augmentation techniques](https://link.springer.com/article/10.1007/s11042-021-11869-x)

TAG_DATA_AUGMENTATION, TAG_NOVEL_DATA, TAG_INSTANCE_SEGMENTATION


1. **Addressing Data Scarcity**: The paper identifies the lack of large datasets as a significant barrier in deep learning applications, specifically in the context of classifying maize crop diseases.
2. **Proposed Augmentation Algorithms**: The authors propose three algorithms: two for acquiring images (from real-world sources and public datasets) and one for applying a wide range of augmentation techniques to enhance dataset size, leading to a dataset increase of up to **1276 times**.
3. **CNN Model Performance**: A new convolutional neural network (CNN) model was developed for classifying four maize crop diseases. The model achieved classification accuracies of **93.39%** on the original dataset and **98.53%** on the augmented dataset, representing a **5.14% improvement** with augmentation.
4. **Comparison with Existing Methods**: The proposed augmentation method outperformed six existing techniques by notable margins in classification accuracy, with improvements of up to **28.31%** over traditional methods like GANs and augmentation frameworks.

Summary
This paper addresses the challenge of limited image datasets in deep learning for classifying maize crop diseases. The authors propose three novel algorithms for image acquisition and a comprehensive image augmentation strategy that applies 52 techniques, significantly enhancing dataset sizes. A new CNN model demonstrates improved classification accuracy with augmented data. The proposed augmentation method outperforms several existing approaches, establishing its effectiveness in improving model performance.



### [The impact of data augmentation and transfer learning on the performance of deep learning models for the segmentation of the hip on 3D magnetic resonance images](https://www.sciencedirect.com/science/article/pii/S2352914823002903)

TAG_DATA_AUGMENTATION

1. **Data Augmentation (DA) vs. Transfer Learning (TL)**: The study finds that data augmentation outperforms transfer learning in improving the accuracy of a CNN for segmenting hip joint structures, particularly when limited datasets are available.
2. **Performance Metrics**: Using data augmentation, the model achieved higher Dice similarity coefficients and accuracy scores for femur and acetabulum segmentation compared to transfer learning.
3. **Segmentation Application**: The model can potentially automate diagnosis, combining with radiomic analysis to aid in evaluating hip pathologies such as femoroacetabular impingement.
4. **Future Directions**: Further work will assess different network architectures to improve segmentation accuracy and explore the relationship between model complexity and estimation accuracy.

Summary
This paper compares the effectiveness of data augmentation and transfer learning for segmenting hip joint structures from MR images using a CNN. Data augmentation significantly improved segmentation accuracy and robustness, proving more effective than transfer learning when data is limited. The model could support automated diagnosis when integrated with radiomic analysis, and future research will explore optimizing network architectures.






---
### Unreal Engine

---

### [UnrealROX: an extremely photorealistic virtual reality environment for robotics simulations and synthetic data generation](https://link.springer.com/article/10.1007/s10055-019-00399-5)

TAG_SYNTHETIC_DATA, TAG_SIM_TO_REAL

1. **Synthetic Data Generation**: UnrealROX, an environment built in Unreal Engine 4, generates high-quality synthetic data with automatic annotations for robotic vision tasks. This approach addresses the limitations of gathering large real-world datasets, offering faster and more efficient data generation.
2. **Photorealism for Reality Gap Reduction**: The photorealistic environments and object interactions in UnrealROX reduce the gap between synthetic and real-world data, enabling better generalization of models trained on synthetic data to real-world applications.
3. **VR-Based Robotic Interaction**: By using virtual reality (VR) setups, like the Oculus Rift or HTC Vive Pro, researchers can control robotic agents in UnrealROX to interact with objects in realistic scenes, enhancing the data’s relevance for robotic vision tasks.
4. **Open-Source and Extensible**: UnrealROX is open-source, allowing researchers to use or extend its tools for various tasks, including semantic segmentation, object detection, and depth estimation.

Summary
UnrealROX provides a VR-based platform for generating synthetic data that narrows the reality gap in robotic vision tasks by incorporating hyperrealistic scenes and interactive robot agents. This environment enables researchers to produce realistic, labeled data efficiently, benefiting tasks like object recognition, segmentation, and navigation. Its open-source nature allows for customization and broader applications in robotic vision research.



### [UnrealGT: Using Unreal Engine to Generate Ground Truth Datasets](https://link.springer.com/chapter/10.1007/978-3-030-33720-9_52)

TAG_SYNTHETIC_DATA

1. **UnrealGT Plugin for Synthetic Data Generation**: UnrealGT is a plugin for Unreal Engine that generates ground truth data for computer vision tasks, such as object detection and visual SLAM, enabling highly configurable synthetic dataset creation.
2. **Comprehensive Data Output**: UnrealGT supports multiple data types—RGB images, depth maps, semantic segmentations, normal maps, and detailed camera parameters—essential for training and evaluating computer vision models.
3. **Customizable and Extensible**: The plugin can enrich scenes with new objects and configurations, allowing users to simulate various conditions (e.g., day/night, seasonal changes) and adapt to specific tasks.
4. **Open-Source Availability**: UnrealGT is freely available under the MIT license on GitHub, offering a flexible tool that leverages Unreal Engine's capabilities for research and community use.

Summary
UnrealGT provides a flexible and powerful solution for generating synthetic ground truth data within Unreal Engine, catering to tasks like object detection and 3D reconstruction. By simulating complex environments and incorporating extensive customization options, UnrealGT serves as a practical alternative to real-world data collection, facilitating advancements in neural network training and evaluation of visual SLAM algorithms. 



### [Experimental Results on Synthetic Data Generation in Unreal Engine 5 for Real-World Object Detection](https://ieeexplore.ieee.org/document/10171761)

TAG_SYNTHETIC_DATA

1. **Synthetic Data Creation with Photogrammetry and Unreal Engine 5**: Real-world objects are transformed into 3D digital replicas using photogrammetry, which are then used in Unreal Engine 5 (UE5) to generate high-quality synthetic images for object detection (OD) training.
2. **Diverse Scene Simulation**: The synthetic data generation in UE5 includes varying angles, backgrounds, and textures, which boosts the diversity of the training dataset and improves model generalization.
3. **Model Evaluation with YOLOv8**: Object detection models, trained with the synthetic data generated in UE5, achieved a strong F1 score of 0.8 in real-world validation, demonstrating the practical effectiveness of the synthetic data.
4. **Cost-Effective Data Generation**: This approach presents a scalable and cost-efficient method for generating large volumes of training data for computer vision tasks, minimizing the need for manual data collection.

Summary
This paper explores a novel approach to synthetic data generation for object detection using Unreal Engine 5. Real objects are digitized with photogrammetry, creating realistic 3D models that are then used in diverse synthetic scenes to train YOLOv8-based OD models. The results show promising real-world performance, suggesting synthetic data in UE5 as a viable, cost-efficient alternative to traditional data collection.



### [Development of a Novel Object Detection System Based on Synthetic Data Generated from Unreal Game Engine](https://www.mdpi.com/2076-3417/12/17/8534)

TAG_SYNTHETIC_DATA

1. **Synthetic Data for Object Detection**: The study successfully demonstrates that a real-world object detection system can be trained entirely on synthetic data generated from Unreal Engine 4, addressing the challenge of requiring large volumes of annotated training data.
2. **Domain Randomization and HDR Scenes**: A domain-randomized environment was implemented to enhance the variability and realism of the dataset. High-dynamic-range (HDR) scenes were used to improve the photorealism of the generated data, aiding in model generalization.
3. **Influence of Asset Quantity**: The paper emphasizes that the number of different object assets in the training dataset significantly impacts the model's generalization performance, with classes represented by a moderate number of assets (5-10) achieving better detection than those with fewer or excessive assets.
4. **Further Improvements Needed**: Although the trained models exhibited good precision and accuracy, there are indications that further refinement is necessary to improve stability and confidence in the inference results. The paper suggests that additional research into optimizing parameters, such as the learning rate and batch size, is essential for closing the reality gap.

Summary
This paper investigates the feasibility of training an object detection system purely on synthetic data using Unreal Engine 4 and the YOLOv5 neural network model. By leveraging domain randomization and HDR imaging, the authors create a robust dataset that aims to generalize well to real-world applications. The results indicate promising performance, highlighting the importance of asset diversity for effective training. The study lays the groundwork for future research to enhance model stability and explore further optimizations.










