# Laboratório 6 - P2: Construindo um Tokenizador BPE e Explorando o WordPiece

Este repositório contém a resolução do Laboratório 6 da disciplina P104 (Full Transformer). O objetivo principal foi implementar o motor básico do algoritmo **Byte Pair Encoding (BPE)** e explorar o funcionamento do tokenizador **WordPiece** utilizando a biblioteca Hugging Face.

## Conteúdo do Repositório

- `bpe_tokenizer.py`: Implementação das Tarefas 1 e 2 (Motor de Frequências e Loop de Fusão).
- `wordpiece_integration.py`: Implementação da Tarefa 3 (Integração com WordPiece/BERT).
- `transformer_model.py`: Modelo base do Transformer (mantido para referência).

## Relatório: O Significado de `##` no WordPiece

No tokenizador WordPiece (utilizado por modelos como o BERT), os sinais de cerquilha (`##`) antes de um token indicam que ele é uma **sub-palavra** que deve ser anexada ao token anterior para formar uma palavra completa. Por exemplo, a palavra "inconstitucionalmente" pode ser segmentada em `['in', '##cons', '##tit', '##uc', '##ional', '##mente']`. 

O uso dessas sub-palavras é fundamental para a eficiência dos modelos de linguagem modernos, pois impede o **travamento do modelo diante de vocabulários desconhecidos**. Em vez de tratar uma palavra rara como um token único "desconhecido" (UNK), o modelo consegue decompô-la em fragmentos menores que ele já conhece. Isso mantém o tamanho do vocabulário gerenciável (geralmente entre 30.000 e 40.000 tokens) e permite que o modelo generalize o significado de palavras novas a partir de seus componentes morfológicos.

## Citação de Uso de IA Generativa

Em conformidade com as instruções do laboratório, declaro que utilizei assistência de IA generativa para a realização desta atividade:

- **Trechos Gerados/Revisados**: A lógica de substituição de strings na função `merge_vocab` (Tarefa 2) foi construída com auxílio de IA para garantir o uso correto de expressões regulares que tratam os espaços entre os símbolos. Além disso, a estrutura do script de integração com a biblioteca `transformers` (Tarefa 3) foi revisada para seguir as melhores práticas de carregamento de modelos pré-treinados.
- **Revisão Humana**: Todo o código foi testado, validado e os resultados foram conferidos manualmente para garantir que as contagens de frequência e as fusões de tokens estivessem de acordo com os requisitos do documento do laboratório.

---
*Este projeto foi desenvolvido como parte das atividades do Instituto de Ensino Superior ICEV.*
