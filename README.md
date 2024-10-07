1. Original Part
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
    - Strukturierung planen
    - Schreiben beginnen



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

> ### FIXME

Inhaltsverzeichnis V1:

- Einleitung
  - Problemstellung (Was ist gegeben? Was soll das Ergebnis sein? Warum/Wofür?)
  - Instanzsegemntirung
  - Bin-Picking
  - Unbekannte Objekte -> Bedeutung, Grund
  - Tiefenbilder (Was sind diese und warum -> mit Papern begründen)
  - YOLACT?
- Hauptteil
  - Datengenerierung
    - Synthetische Datengenerierung sinnvoll? (nimm Bezug auf Paper)
    - Wie viele Daten? 
    - Wie wichtig ist die Qualität? 
    - Welche Daten werden benötigt?
    - Wie wurden die Daten generiert?
    - Erstellung welcher Datensätze?
    - Vor- und Nachteile (Labeling, Qualität, ...)
    - Ergebnis der Datengenerierung
  - Instanzsegmentierung mit YOLACT
    - Was wurde hier vorbereitet, gemacht?
    - Mask RCNN?
    - Training mit den Datensätzen
    - Welche Model-Versionen werden analysiert
  - Experiment: Reales Bin-Picking
    - Aufbau des Experiments (es müssen neue/unbekannte Objekte sein, es muss objektiv bewertbar sein)
    - Wie das Bin-Picking funktioniert
    - Ergebnisse (ohne Wertung)
      - Allgemeines Ergebnis des Bin-Pickings
      - Ergebnis der Segmentierung an sich:
        - Accuracy? 
        - Bildvergleich
        - ...
  - Performance auf dem COCO-Datensatz -> hat dieser Tiefenbilder? Wenn nicht, ist er uninteressant
    - Ergebnisse
    - Warum sich der COCO-Datensatz nicht für Instanzsegmentierung eignet (evtl. Quellen hierfür?)
  - Performance auf dem WISDOM-Datensatz 
    - Warum WISDOM?
    - Ergebnisse auf real
    - Ergebnisse auf Synth
  - Gibt es noch einen weiteren bekannten Tiefenbilddatensatz für Segmentierung?
- Resumee (Vergleich)
  - Wie sind die Ergebnisse zu deuten?
  - Gab es einen signifikanten Unterschied?
  - Warum könnten die Ergebnisse so sein, wie sie sind? Was könnte man daraus schlussfolgern?
  - Hat das Sim-zu-Real geklappt?
  - Welche offenen Fragen ergeben sich bzw. wären diesbezüglich noch interessant?




```text
Inhaltsverzeichnis V2:

1. Hintergrund und Motivation
    1.1 Problemstellung der Instance-Segmentierung im Bin-picking
    1.2 Ziel und Relevanz der Untersuchung von Material- und 3D-Modellkombinationen
    1.3 Aufbau der Arbeit

2.Theoretische Grundlagen und Stand der Technik
    2.1 Grundlagen der Instance-Segmentierung
    2.1.1 Definition und Funktionsweise
    2.1.2 Anwendungsfall: Bin-picking in der Robotik
    2.2 Machine Learning und Deep Learning Algorithmen für die Segmentierung
    2.2.1 Convolutional Neural Networks und ihre Rolle in der Segmentierung
    2.3 YOLACT-Algorithmus
    2.3.1 Architektur und Anpassungen für Bin-picking
    2.3.2 Wrapper und Modifikationen für dieses Projekt
    2.4 Unreal Engine 5 für synthetische Datengenerierung
    2.4.1 Erstellung von 3D-Modelldaten
    2.4.2 Automatisierte Materialgenerierung und -zuweisung

3. Verwandte Arbeiten und Herausforderungen bei Sim-to-Real-Transfer
    3.1 Stand der Forschung zu Material- und Formeinflüssen auf Segmentierungsalgorithmen
    3.2 Herausforderungen beim Sim-to-Real Transfer in der Robotik
    3.3 Relevante Datensätze und Benchmarks (z.B. OCID Dataset)

4. Methodik zur Datenerstellung und Testung
    4.1 Auswahl und Erstellung der 3D-Modelle
    4.1.1 Manuelle Auswahl: Kriterien und Ausschluss von Modellen
    4.1.2 Subjektive Unterscheidung in Form und Struktur
    4.2 Materialfilterung mittels SSIM-Index
    4.2.1 Vermeidung ähnlicher Materialien
    4.2.2 Beispiele für gefilterte Materialien
    4.3 Datensatzgenerierung mit Unreal Engine 5
    4.3.1 Prozess zur Kombination von Materialien und Modellen
    4.3.2 Faktoren zur Variierung im Datensatz
    4.3.2.1 Anzahl der Objekte
    4.3.2.2 Position der Objekte
    4.3.2.3 Material der Bin-Box
    4.3.2.4 Material des Bodens
    4.3.2.5 Kameraposition

5. Experimente und Testdesign
    5.1 Überblick über die Datensätze
    5.1.1 OCID-Datensatz: Evaluierung der Sim-to-Real-Fähigkeit
    5.1.2 Synthetischer Datensatz: Generiert mit Unreal Engine 5
    5.1.3 Grenzfalldatensatz: Untersuchung von Material- vs. Formabhängigkeit
    5.2 Evaluationsmethoden
    5.2.1 Mean Average Precision (mAP) und Intersection over Union (IoU)
    5.2.2 Bias-Analyse: Shape-Bias vs. Material-Bias

6. Ergebnisse und Diskussion
    6.1 Ergebnisse der Sim-to-Real-Überprüfung
    6.1.1 Leistung auf dem OCID-Datensatz
    6.1.2 Sim-to-Real-Verluste und Herausforderungen
    6.2 Ergebnisse auf synthetischen Datensätzen
    6.2.1 Performance im Vergleich zwischen Unreal Engine 5 generierten Daten und Grenzfällen
    6.3 Diskussion der Bias-Untersuchungen
    6.3.1 Shape-Bias: Einfluss der Form auf die Genauigkeit
    6.3.2 Material-Bias: Einfluss der Materialbeschaffenheit auf die Genauigkeit
    6.4 Vergleich mit verwandten Arbeiten

7. Fazit und Ausblick
    7.1 Zusammenfassung der Ergebnisse und Implikationen
    7.2 Beitrag der Arbeit zur Forschung im Bereich Bin-picking und Instance-Segmentierung
    7.3 Ausblick auf zukünftige Arbeiten
    7.3.1 Erweiterung der Datensätze
    7.3.2 Optimierungspotential bei der Sim-to-Real-Übertragung

8. Anhang
    8.1 Quellcode des Unreal Engine Programms
    8.2 Quellcode des YOLACT Wrappers und der Anpassungen
    8.3 Zusätzliche Experimente und Analysen
    8.4 Datensatzbeschreibung

9. Literaturverzeichnis
```



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



### Quellen

Bildgröße:
...

Datensatz größe:
...

Sim-to-real:
...

Shape-Material Bias:
- [Can Biases in ImageNet Models Explain Generalization?](https://openaccess.thecvf.com/content/CVPR2024/papers/Gavrikov_Can_Biases_in_ImageNet_Models_Explain_Generalization_CVPR_2024_paper.pdf)
- [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231)



### Präsentation

Struktur:

- Thema, Problemstellung (Motivation, Relevanz)
- YOLACT Vorstellung
- Datengenerierung
- Experiment
- Aktuelle Ergebnisse + Schlussfolgerung (Mehrwert)
- Fazit




