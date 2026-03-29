import re
from collections import defaultdict

def get_stats(vocab):
    """
    Tarefa 1: O Motor de Frequências
    Calcula a frequência de todos os pares adjacentes de caracteres/símbolos.
    """
    pairs = defaultdict(int)
    for word, frequency in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += frequency
    return pairs

def merge_vocab(pair, v_in):
    """
    Tarefa 2: O Loop de Fusão
    Substitui todas as ocorrências do par isolado pela versão unificada.
    """
    v_out = {}
    # Escapa caracteres especiais como '/' em </w>
    bigram = re.escape(' '.join(pair))
    # Substitui o par (ex: 'e s') pela versão unificada (ex: 'es')
    # Usamos lookbehind e lookahead para garantir que estamos pegando o par isolado
    # mas o BPE trabalha com espaços entre os símbolos, então uma substituição simples
    # de string pode ser suficiente se tivermos cuidado com os espaços.
    
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    replacement = ''.join(pair)
    
    for word in v_in:
        w_out = p.sub(replacement, word)
        v_out[w_out] = v_in[word]
    return v_out

def main():
    # Inicialização do vocabulário conforme o documento
    # Nota: As palavras no BPE são representadas como sequências de caracteres separadas por espaços
    vocab = {
        'l o w </w>': 5,
        'l o w e r </w>': 2,
        'n e w e s t </w>': 6,
        'w i d e s t </w>': 3
    }

    print("Vocabulário Inicial:")
    for word, freq in vocab.items():
        print(f"'{word}': {freq}")
    print("-" * 30)

    # Validação da Tarefa 1
    stats = get_stats(vocab)
    print("Frequência dos Pares (Tarefa 1):")
    # Ordenar por frequência para facilitar visualização
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    for pair, freq in sorted_stats:
        print(f"{pair}: {freq}")
    
    if stats[('e', 's')] == 9:
        print("\nValidação Tarefa 1: SUCESSO (Par ('e', 's') tem contagem 9)")
    else:
        print(f"\nValidação Tarefa 1: FALHA (Par ('e', 's') tem contagem {stats.get(('e', 's'), 0)}, esperado 9)")
    
    print("-" * 30)

    # Tarefa 2: Loop de Treinamento (5 iterações)
    num_merges = 5
    current_vocab = vocab.copy()
    for i in range(num_merges):
        stats = get_stats(current_vocab)
        if not stats:
            print("Não há mais pares para fundir.")
            break
        best = max(stats, key=stats.get)
        current_vocab = merge_vocab(best, current_vocab)
        print(f"Iteração {i + 1}:")
        print(f"  Par mais frequente fundido: {best} (frequência {stats[best]})")
        print(f"  Estado do vocabulário:")
        for word, freq in current_vocab.items():
            print(f"    '{word}': {freq}")
        print("-" * 20)

    # Validação final da Tarefa 2
    print("\nResultado Final após 5 iterações:")
    for word in current_vocab:
        print(f"'{word}'")
    
    # Verificar se 'est</w>' foi formado (como mencionado na validação do PDF)
    found_est = any('est</w>' in word for word in current_vocab)
    if found_est:
        print("\nValidação Tarefa 2: SUCESSO (Token 'est</w>' ou similar foi formado)")
    else:
        print("\nValidação Tarefa 2: AVISO (Verifique se os tokens morfológicos foram formados conforme esperado)")

if __name__ == "__main__":
    main()
