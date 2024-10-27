It follows:
1. Original Part -> https://github.com/xXAI-botXx/Vorlage-Abschlussarbeit-Latex
2. My notes

# LaTeX-Vorlage für Abschlussarbeiten 

Die Datei `thesis.tex` ist die Hauptdatei des Projekts.  
In der Datei `docinfo.tex` können die Angaben zu Titel, Autor etc. eingetragen
werden.  
Das Editieren der Datei `titelblatt.tex` ist normalerweise nicht notwendig.
Lediglich wenn auch das Logo eines Unternehmens auf dem Titelblatt abgebildet
werden soll, muss die Datei angepasst werden. Ein Beispiel dazu ist als
Kommentar in der Datei vorhanden. 

Als Compiler ist [Latexmk](http://personal.psu.edu/~jcc8/software/latexmk/) zu
verwenden. Siehe Datei `Makefile`.

Die Vorlage wurde von Andreas Männle zur Verfügung gestellt. Vielen Dank!

Anregungen und Hinweise zur Optimierung der Vorlage senden Sie an
stefan.hensel@hs-offenburg.de

Viel Erfolg!


---

# Meine Bachelorarbeit

Dieses Dokument dient mir als Begleitdokument bei der Erstellung meiner Bachelorarbeit. Hier werde ich alle relevanten Notizen und Anmerkungen und Planungen machen.

Von Tobia Ippolito 2024 bei dem Unternehmen Optonic GmbH in Freiburg.

Tags: Computer Vision, Deep Learning, Syntethic Datageneration

Sprache: Englisch

Title:
- Depth Data and Shape-Texture Biases in Instance Segmentation
or
- Using 3D-Information Improves Shape Awareness



### Schreibplan

- Schreibumgebung zum Laufen bekommen (Word von der HS)
- Quellen zum Thema Sammeln -> evtl. Zusammenfassungen erstellen oder erstellen lassen
- Struktur (Inhaltsverzeichnis schreiben)
- Abstract schreiben (eventuell erst am Ende)
- ...


### Installation

Ich empfehle es auf Linux zu installieren:
```terminal
sudo apt update
sudo apt install latexmk
sudo apt install texlive-full
sudo apt install texstudio
```



### Gesamt-Durchführung

1. Deep-learning Model für Instanz-Segmentierung zum Laufen bekommen
    - Auswahl des Models
    - Einsatzbereit programmieren
2. Datensatz für Model
    - 3D-Modelle finden und in ein standardisiertes Format bringen
    - Materialien finden und in ein standardisiertes Format bringen
    - Synthetische Datengenerierung programmieren (Unreal Engine?)
    - Datensatz anhand von geplanten Experimenten erstellen
    - Testdatensätze bereitstellen
3. Training + Evaluierung
    - Definieren von Trainingsvariablen (Epochen, Lernrate, Datenmenge, ...)
    - Models für jeden Datensatz erstellen
    - Geeigneten Test-Datensatz finden
    - Evaluieren der Ergebnisse anhand eines Test-Datensatzes
    - Finentuning auf einem neuen Trainingsdatensatz? Fürs Benchmarking?
    - Was heißt das Ergbnis? Was könnte es zuküngtig bedeuten? In Bezug zu anderen Ergbnissen setzen.
4. Bachelor schreiben
    - Quellen finden
    - Strukturierung planen (Inhaltsverzeichnis)
    - Einleitung schreiben
    - Grundlagen und Definitionen Kapitel schreiben
    - Hauptteil schreiben
    - Schluss schreiben
    - Abstract schreiben
    - Korrektur lesen und schreiben



### Thema

In meiner Bachelorarbeit möchte ich herausfinden, wie viel unterschiedliche 3D-Modelle und wie viel unterschiedliche Materialien es in den Trainingsdaten für die Instanzsegmentierung im Gebiet des Bin-Pickings mit RGB Bildern und YOLACT als AI-Model benötigt wird, um eine hohe Accuracy zu erlangen.

Außerdem möchte ich herausfinden, ob ein Deep-Learning Model generell genauer wird, desto mehr Materialien beziehungsweise desto mehr 3D-Modelle es in den Daten gibt.

Das Ergebnis könnte zeigen, dass sich KI-Modelle eher auf Shapes oder eher auf die RGB-Daten (Materialien) konzentrieren (einen Bias dahingegend haben). Hierzu möchte ich spezielle Grenzbeispiele erstellen, bei welchen unklar ist, ob man sich nun auf das Material stützt oder auf die Shape, um den Bias deutlicher verifizieren zu können.

Schlussendlich werden meine Experimente eventuell noch Aufschluss auf die Güte des Transfers von der Simulation zur Realität in Bezug auf die Anzahl der 3D-Modelle und Materialien geben.




### Relevanz

Trainingsdaten sind im Bereich Deep-Learning immer eines der größten Herausforderungen. Gerade im Bereich Segmentierung kann das Erstellen eines Datensatzes herausfordern sein, da jeder Pixel für die Lösungsmaske (Label) benötigt wird. Dies per hand zu labeln würde sehr viel Zeit beanspruchen und so gibt es sehr viele Arbeiten, welche auf syntethisch generierte Daten setzt (siehe ...).
...



### Inhaltsverzeichnis

> Klassische Begriffe, wie Einleitung bitte spzifizieren und ersetzen + der Leser sollte Wissen können was ihn grob erwartet
> Vom Überthema zum Unterthema sollte sich immer mehr relevante Details binhaltet sein 

Grundlegende Struktur:
```text
0. Abstract
    A short summary of the paper. It spoils you directly and gives you the answer, what this paper analyzed and what it founds

1. Introduction
    (1) Why is my work/question important and interesting?
    (2) What is the concrete question and problem? Which scientific question will be tried to answer? Why/how is there a knowledge gap?
    (3) How did I analyzed/answered this question?
    (4) Delimitation of my work. What will be not answered? (And why?) What will be answered?
    (5) How will you answer this question? What follows in the next chapters?

2. Definitions and Basics


3. Main


4. End


```

Inhaltsverzeichnis verbessert: -> etwa 10 Seiten pro Kapitel (außer Fazit)

- Einleitung -> Thema einleitung, Systematik, verwandte Arbeiten
- Implementierung -> Tools, Umgebung, Womit wird gearbeitet
- (Messungen +) Versuchsbeschreibung
- Versuchsergebnis
- Fazit -> Interpretation + Ausblick



Dramaturgischer Bogen -> "Wie ein Roman"


Table of Contents in your context (not finish version)

Scientific Question: 

- Investigation of Material and Shape Quantities on Instance Segmentation Accuracy
  - Subquestion: Assessing Bias Toward Material or Shape in Instance Segmentation (maybe you have to put more materials than shapes to decrease this bias)
    -> When the accuracy is better with more Materials than maybe it means that there is a Shape-Bias
    -> Maybe let this question open? Or you could make another experiment with special Shapes and Materials to get a picture about it -> Which Models decides for Shape and which for Material?
  - Subquestion: Also in relation to Sim-to-real accuracy

Delimitations:
- Just for CNN Based Deep Learning Approaches -> Cnn-Based Methods most widely used for Instance-Segmentation (source?)
- Only for Instance-Segmentation -> Because there is a lack of datasets?
- The choose of the random shape and material could also have an influence on the result 
- Only on RGB images -> RGB matters + depth no materials
- Specific for Bin Picking?
- Testing on Unknown Data -> Unknown meshes and unknown materials -> to reduce the influence from other sources (o.o.d scenarios)

Experiments:
- YOLACT trained on different 3xM_Datasets:
  - Syntethic Testdataset (maybe own 3xM Testdataset, but with unknown 3D-Models and Materials -> so that there is no influence from the novel-objects -> for all models the data are unknown, else for some there will be a subpart of materials and shapes known which leads to another influence)
  - Real Testdataset
  - Outlier-Data


Einfluss:
- Some Shapes and Materials are more comple than others, and so the choice of which of them will be choosen is important and could lead to another result. How big this influence really is is unclear but could be tested in future. Questionable is also if it is more benefitial to have more or less complex shapes and materials.


```
0. Abstract -> not knowing yet

1. FIXMe


```



### Namenskonvention

- Shape = 3D-Model, Mesh = The 3D geometry of an object
- Material = Texture = The visual characteristics of an object (like Color, Metalness, Roughness, ...)

FIXME


### Experimentenplan


**Trainingsdauer:**

Dauer von einer Datengenerierung ~= 24 Stunden
Dauer von einem Training ~= 12 Stunden

Anzahl an Materialien/3D-Modellen = [1, 10, 100, 200]

Anzahl an Datensätzen = Anzahl an Materialien * Anzahl an 3D-Modellen
Anzahl an Datensätzen = 4 * 4 * 24 = 16 Tage

Gesamtdauer = Anzahl_Experimente * Anzahl an Datensätzen * Dauer von einem Training
...

> Es muss noch ein Puffer eingerechnet werden, da zwischen den Trainingseinheiten evtl. ein zetlicher Puffer entsteht.
> Durch das parallele Erstellen und Trainieren, entsteht keine Reduzierung der Zeit durch das Erstellen der Datensätze.


**Ziel Ergebnisse:**

Evaluation auf folgende Datensätzen:
- [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/)
  - Reale Daten
  - Unknown/Novel Objects
- Eigener Testdatensatz
  - Synthetische Daten
  - Known Objects (je nach Model)
- COCO -> nur wenn genug Zeit, eig unnötig
  - Reale Daten
  - "Schlecht" für Instanzsegmentierung
  - Unknown Objects

Evaluation mit folgenden KI-Modellen:
- 3xM YOLACT 1-1
- 3xM YOLACT 1-10
- 3xM YOLACT 1-100
- 3xM YOLACT 1-200
- 3xM YOLACT 10-1
- 3xM YOLACT 100-1
- 3xM YOLACT 200-1
- 3xM YOLACT 10-10
- 3xM YOLACT 100-10
- 3xM YOLACT 200-10
- ...
- COCO YOLACT
- WISDOM YOLACT? -> keine Tiefenbilder


Metriken:
- Pixel-Accuracy
- ...


### Präsentation

Struktur:

- Thema, Problemstellung (Motivation, Relevanz)
- YOLACT Vorstellung
- Datengenerierung
- Experiment
- Aktuelle Ergebnisse + Schlussfolgerung (Mehrwert)
- Fazit


### Quellen & Fakten

[Klicke hier, um meine Quellen und derren Zusammenfassungen zu sehen.](./Source.md)

Fakten:
- CNN-Methods lack to learn global information and prefers to learn local information 
  - [Deep convolutional networks do not classify based on global object shape](./Source.md#deep-convolutional-networks-do-not-classify-based-on-global-object-shape)
- CNN have an Texture bias 
  - [Deep convolutional networks do not classify based on global object shape](./Source.md#deep-convolutional-networks-do-not-classify-based-on-global-object-shape)
  - [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](./Source.md#imagenet-trained-cnns-are-biased-towards-texture-increasing-shape-bias-improves-accuracy-and-robustness)
  - [Shape or Texture: Understanding Discriminative Features in CNNs](./Source.md#shape-or-texture-understanding-discriminative-features-in-cnns)
  - [Trapped in Texture Bias? A Large Scale Comparison of Deep Instance Segmentation](./Source.md#trapped-in-texture-bias-a-large-scale-comparison-of-deep-instance-segmentation)
- Depth Images are popular because the successfull sim-to-real
  - [Segmenting Unknown 3D Objects from Real Depth Images using Mask R-CNN Trained on Synthetic Data](./Source.md#segmenting-unknown-3d-objects-from-real-depth-images-using-mask-r-cnn-trained-on-synthetic-data)
- Recently RGB only data seams to have also an high performance performance
  - [Bin-picking of novel objects through category-agnostic-segmentation: RGB matters](./Source.md#bin-picking-of-novel-objects-through-category-agnostic-segmentation-rgb-matters)
- CNN's are only Texture Biased, because it is the more easy information
  - [Towards Synthetic Data: Dealing with the Texture-Bias in Sim2real Learning](./Source.md#towards-synthetic-data-dealing-with-the-texture-bias-in-sim2real-learning)
  - [A Competition of Shape and Texture Bias by Multi-view Image Representation](./Source.md#a-competition-of-shape-and-texture-bias-by-multi-view-image-representation)
- Shape still playes a role in CNN's
  - [Shape or Texture: Understanding Discriminative Features in CNNs](./Source.md#shape-or-texture-understanding-discriminative-features-in-cnns)
  - [Towards Synthetic Data: Dealing with the Texture-Bias in Sim2real Learning](./Source.md#towards-synthetic-data-dealing-with-the-texture-bias-in-sim2real-learning)
- High Resolution Textures helping bridging the gap of sim-to-real
  - [Towards Synthetic Data: Dealing with the Texture-Bias in Sim2real Learning](./Source.md#towards-synthetic-data-dealing-with-the-texture-bias-in-sim2real-learning)
- Texture Bias alone can't explain the generalization of a model
  - [Can Biases in ImageNet Models Explain Generalization?](./Source.md#can-biases-in-imagenet-models-explain-generalization)
  - [Harnessing Synthetic Datasets: The Role of Shape Bias in Deep Neural Network Generalization](./Source.md#harnessing-synthetic-datasets-the-role-of-shape-bias-in-deep-neural-network-generalization)
- Texture Bias come also from the data itself
  - [The Origins and Prevalence of Texture Bias in Convolutional Neural Networks](./Source.md#the-origins-and-prevalence-of-texture-bias-in-convolutional-neural-networks)
- Shape-Bias increase the performance of deep learning models
  - [The Origins and Prevalence of Texture Bias in Convolutional Neural Networks](./Source.md#the-origins-and-prevalence-of-texture-bias-in-convolutional-neural-networks)
- In OOD (Out of Distribution) datasets, shape bias is not always better
  - [Shape-biased CNNs are Not Always Superior in Out-of-Distribution Robustness](./Source.md#shape-biased-cnns-are-not-always-superior-in-out-of-distribution-robustness)
- Shape-Texture debiased models have a higher accuracy
  - [SHAPE-TEXTURE DEBIASED NEURAL NETWORK TRAINING](./Source.md#shape-texture-debiased-neural-network-training)
- The combination of shape biased and texture biased models leads to better results (with ensemble method)
  - [UNIVERSAL ADVERSARIAL ROBUSTNESS OF TEXTURE AND SHAPE-BIASED MODELS](./Source.md#universal-adversarial-robustness-of-texture-and-shape-biased-models)
  - [Shape Prior is Not All You Need: Discovering Balance Between Texture and Shape Bias in CNN](./Source.md#shape-prior-is-not-all-you-need-discovering-balance-between-texture-and-shape-bias-in-cnn)
  -> it is another approach to mine
- Data causes Texture bias
  - [A Competition of Shape and Texture Bias by Multi-view Image Representation](./Source.md#a-competition-of-shape-and-texture-bias-by-multi-view-image-representation)
- Mask-RCNN is still a very well performaning instance-segmention method
- Reducing Texture Bias can be helpful
  - [On the Texture Bias for Few-Shot CNN Segmentation](./Source.md#on-the-texture-bias-for-few-shot-cnn-segmentation)
- Not the Shape-Complexity is important, else the combination of shape and color is very important for good results and learning
  - [On the Influence of Shape, Texture and Color for Learning Semantic Segmentation](./Source.md#on-the-influence-of-shape-texture-and-color-for-learning-semantic-segmentation)
- Depth Images are improving segmentation
  - [Benchmarking of deep learning algorithms for 3D instance segmentation of confocal image datasets](./Source.md#benchmarking-of-deep-learning-algorithms-for-3d-instance-segmentation-of-confocal-image-datasets)
  - [Exploiting Depth Information for Wildlife Monitoring](./Source.md#exploiting-depth-information-for-wildlife-monitoring)
  - [Depth-aware object instance segmentation]()
- Instance Segmentation is widely used
- Synthetic Data is very promising for depp learning
  - [Driving in the Matrix: Can Virtual Worlds Replace Human-Generated Annotations for Real World Tasks?](./Source.md#driving-in-the-matrix-can-virtual-worlds-replace-human-generated-annotations-for-real-world-tasks)
  - [The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes](./Source.md#the-synthia-dataset-a-large-collection-of-synthetic-images-for-semantic-segmentation-of-urban-scenes)
- Synthetic Data has often also a lack of quality
  - [Harnessing Synthetic Datasets: The Role of Shape Bias in Deep Neural Network Generalization](./Source.md#harnessing-synthetic-datasets-the-role-of-shape-bias-in-deep-neural-network-generalization)
- Data Augmentation does improve the performance and the ability for generalization
  - [Data Augmentation in Classification and Segmentation: A Survey and New Strategies](./Source.md#data-augmentation-in-classification-and-segmentation-a-survey-and-new-strategies)
  - [Image- and Instance-Level Data Augmentation for Occluded Instance Segmentation](./Source.md#image--and-instance-level-data-augmentation-for-occluded-instance-segmentation) -> helpful for not much data, which is a common problem in the field of instance segmentation
  - [Data Augmentation Methods for Semantic Segmentation-based Mobile Robot Perception System](./Source.md#data-augmentation-methods-for-semantic-segmentation-based-mobile-robot-perception-system)
  - [Data augmentation: A comprehensive survey of modern approaches](./Source.md#data-augmentation-a-comprehensive-survey-of-modern-approaches)
  - [Performance improvement of Deep Learning Models using image augmentation techniques](./Source.md#performance-improvement-of-deep-learning-models-using-image-augmentation-techniques)
  - [The impact of data augmentation and transfer learning on the performance of deep learning models for the segmentation of the hip on 3D magnetic resonance images](./Source.md#the-impact-of-data-augmentation-and-transfer-learning-on-the-performance-of-deep-learning-models-for-the-segmentation-of-the-hip-on-3d-magnetic-resonance-images)
- Collecting high quality annotated data is difficult
  - [Data augmentation: A comprehensive survey of modern approaches](./Source.md#data-augmentation-a-comprehensive-survey-of-modern-approaches)
  - [Image- and Instance-Level Data Augmentation for Occluded Instance Segmentation](./Source.md#image--and-instance-level-data-augmentation-for-occluded-instance-segmentation)
- using Unreal Engine for generating synth data
  - [UnrealROX: an extremely photorealistic virtual reality environment for robotics simulations and synthetic data generation](./Source.md#unrealrox-an-extremely-photorealistic-virtual-reality-environment-for-robotics-simulations-and-synthetic-data-generation)
  - [UnrealGT: Using Unreal Engine to Generate Ground Truth Datasets](./Source.md#unrealgt-using-unreal-engine-to-generate-ground-truth-datasets)
  - [Experimental Results on Synthetic Data Generation in Unreal Engine 5 for Real-World Object Detection](./Source.md#experimental-results-on-synthetic-data-generation-in-unreal-engine-5-for-real-world-object-detection)
  - [Development of a Novel Object Detection System Based on Synthetic Data Generated from Unreal Game Engine](./Source.md#development-of-a-novel-object-detection-system-based-on-synthetic-data-generated-from-unreal-game-engine)


FIXME



### Weiteres

o.o.d (Out of Distribution) can be:
- Image Corruption: This includes noise, blur, or distortions in images that were not present during training. For example, a model trained on clear, high-resolution images might struggle when presented with blurry or pixelated versions of those images.

- Adversarial Attacks: In this context, specially crafted inputs are designed to deceive models, leading them to make incorrect predictions. These adversarial examples often lie outside the distribution of training data.

- Domain Shift: This occurs when there is a change in the data collection environment or methodology. For example, a model trained on images taken in bright daylight may perform poorly on images taken at night or in different lighting conditions.

- Style Shift: When the aesthetic or stylistic properties of the input data change, models might struggle. For instance, a model trained on photographs may not perform well on artistic images or sketches.

- Dataset Shift: This includes any changes in the underlying distribution of data points, such as changes in class distributions or features not present during training.


---


Ja, das klingt nach einer Untersuchung des Shape-Texture-Bias. Du testest, wie gut dein Modell in der Lage ist, zwischen Formen und Texturen zu unterscheiden und ob es dazu neigt, die Segmentierung auf der Basis der Form oder der Textur durchzuführen.

Hier sind einige spezifische Aspekte, die du untersuchst:

1. Textursegmentierung: Du prüfst, ob das Modell besser in der Lage ist, Objekte anhand ihrer Texturen zu segmentieren, selbst wenn diese Texturen auf unterschiedlichen 3D-Modellen angewendet werden.


2. Formsegmentierung: Du untersuchst, ob das Modell dazu neigt, Objekte hauptsächlich anhand ihrer Form zu segmentieren, auch wenn verschiedene Texturen verwendet werden.


3. Kombination von Form und Textur: Du kannst auch testen, wie das Modell bei Objekten abschneidet, die sowohl verschiedene Formen als auch Texturen kombinieren.


4. Bias-Analyse: Indem du die Genauigkeit und Performance deines Modells in diesen Szenarien analysierst, kannst du herausfinden, ob ein Bias in Bezug auf Form oder Textur besteht.


Zusammenfassend lässt sich sagen, dass du den Shape-Texture-Bias untersuchst, um zu verstehen, welche Merkmale dein Modell stärker gewichtet, wenn es um die Segmentierung von Objekten geht.








