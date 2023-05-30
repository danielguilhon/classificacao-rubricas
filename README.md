# Projetos da Fiscalização Contínua de Folha de Pagamentos - FCP

Todos os trabalhos relativos à FCP ficarão nesse Repo
## Criando o ambiente conda
```
conda create -n fcp python=3.11 pandas scikit-learn xgboost jupyter sqlalchemy cx_Oracle pyodbc pyarrow nltk unidecode

conda activate fcp
```
## Clonando o Repositório

- Clonar o repositório usando os comandos abaixo:
```
cd <<raiz_onde_vai_residir_repo>
git clone http://git.tcu.gov.br/sefip/fcp.git
```

## Instalando o FastText

- Clonar o repositório Git (https://github.com/facebookresearch/fastText) da versão mais recente
- Entra no diretório
- Executa pip install .
```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
```
- Testa a instalação importando o módulo:
```
import fasttext
```

## Configurando o NLTK
### Verificar se já está baixado o Stemmer em portugues (RSLP)
- Em uma janela de terminal, executa o python e roda o downloader
```
python
import nltk 
nltk.download()
```

## Desenvolvimento e Manutenção
- Recomenda-se o desenvolvimento e manutenção do código utilizando o VSCode disponível no LabContas, com o módulo Remote SSH, acessando atualmente o servidor srv-rstudio
- Após alterações de código, realizar o 'stage' e 'commit' no VSCode
- Entrar via terminal no srv-rstudio e fazer um "git pull" para subir as alterações de código para o Repo.

## Autores
### AudPessoal - SIA
- Daniel Guilhon
- Leandro Grapiúna
