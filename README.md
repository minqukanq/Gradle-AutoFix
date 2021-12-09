# Gradle-AutoFix: An Automatic Resolution Generator For Gradle Build Error

## Abstract
Gradle is one of the widely used tools to automatically build a software project. While
developers execute the Gradle build for projects, they face various build errors in practice.
However, fixing build errors is not easy because developers manually should find out the
cause of the build error and its resolution on their project. This problem makes developers
spend a lot of time. In particular, it can be worse if a developer lacks the experience
of handling build errors. To address this issue, we propose a novel approach named
Gradle-AutoFix to automatically fixing build errors with provided resolutions and their
causes. In this approach, we collect build errors to group their causes and resolutions and
then generate feature vectors from build error messages by utilizing Bag-of-Word (Bow),
TF-IDF, Bigram, and an embedding layer. Feature vectors are trained to build two
classification models for providing cause and resolution. Next, we analyze fixing patterns
and define seven resolution reules to automatically fix them on the project. Based on
models and resolution rules, we built Gradle-Autofix
