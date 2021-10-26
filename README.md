# ASR project barebones

## Installation guide

Перед запуском установите зависимости.

```shell
pip install -r ./requirements.txt
```
В случае, если ctcdecode не устанавливается, воспользуйтесь инструкцией в README исходном репозитории: https://github.com/parlance/ctcdecode

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```
(лучше сделать это снаружи корневой директории)

Финальная модель обучалась в 4 этапа, их конфиги лежат в
[директории](https://github.com/ivan-gorin/asr_project_template/tree/hw_asr_2021/hw_asr/configs/final_configs)

