# EnsembleBasedTSC
Project based on the paper Deep Neural Network Ensembles for Time Series Classification (2019) by Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, Pierre-Alain Muller
:)


## TODO:
- [X] Simplify Notebooks by creating functions that are used by everyone -> Felix
- [X] Use glorot initialization for the weights (see https://keras.io/api/layers/initializers/) -> Felix
- [-] Aufschreiben, welche Datensätze komisch sind und entscheiden, welche wir benutzen -> Tanja
- [ ] Ensembles evaluieren -> Felix
- [-] Transfer Learning für FCN -> anders implementieren; Funktion die ein Dataset für Tuning, eins für Finetuning, und Model erhält  -> Tim
- [ ] MCDCNN und Time-CNN verbessern/tunen/linearisieren/batch norm -> Tim
- [ ] Vergleich SGD vs Adam -> Tanja
- [ ] Confidence der Vorhersagen als Confidence der Ensembles -> ???
- [x] Confusion Matrix -> Tim
- [x] Tim: MCDCNN, Time-CNN -> Tim
- [x] Felix: ResNet, FCN -> Felix
- [x] Tanja: MLP, Encoder -> Tanja
- [x] Datensätze verteilen an alle -> Tanja
- [x] Notebook das **ein** Modell trainiert und evaluiert -> Tanja
- [x] Herausfinden, warum Training der Netze teilweise nicht funktioniert -> Alle

- [ ] Zukünftig: RNN, LSTM -> Tim
- [ ] LSTM,RNN für Transfer Learning? -> Tim
- [ ] Zukünftig: Perturbation of Datasets? -> Tanja
- [ ] Zukünftig: float16 statt float32? Performance Vergleich?
- [ ] Zukünftig: Vorverarbeiten der Daten (z.B. Fourier Transformation)


Allgemein: Zeitnah Modelle vollständig trainieren und speichern 

Anmerkung: Während dem Programmieren Notizen für die Seminararbeit machen
