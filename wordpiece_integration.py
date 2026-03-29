from transformers import AutoTokenizer

def main():
    """
    Tarefa 3: Integração Industrial e WordPiece
    """
    # 1. Instanciar o tokenizador multilíngue do BERT (WordPiece)
    print("Carregando o tokenizador 'bert-base-multilingual-cased'...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # 2. Frase de teste para forçar o particionamento morfológico
    test_sentence = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."

    # 3. Utilizar o método .tokenize() para segmentar a frase
    tokens = tokenizer.tokenize(test_sentence)

    # 4. Imprimir o resultado no terminal
    print("\nFrase de teste:")
    print(test_sentence)
    print("\nTokens resultantes (WordPiece):")
    print(tokens)

    # Exemplo de como o WordPiece lida com palavras desconhecidas ou longas
    print("\nExplicação sobre os sinais de cerquilha (##):")
    print("Os sinais '##' indicam que o token é uma sub-palavra (continuação) de um token anterior.")
    print("Isso permite que o modelo represente qualquer palavra como uma sequência de sub-palavras conhecidas,")
    print("evitando o problema de 'vocabulário desconhecido' (Out-of-Vocabulary - OOV).")

if __name__ == "__main__":
    main()
