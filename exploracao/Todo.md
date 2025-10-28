# 30/9/2025

## Resumo
- Tive a ver alguns artigos relacionados com usar algoritmos evolutivos para optimizar LLMs
- Excel: https://docs.google.com/spreadsheets/d/1z3uQhLXzJZl6gbDWk_YRzllY7TfAK0FZrW3-iZ3KoH4/edit?gid=0#gid=0
- Coisas que me chamaram a atenção: Os algoritmos de evolução podem ser usados em tarefas diferentes (Model merging, Prompt tuning, hyperparameters), mas
segundo o que li prompt tuning é onde se encontrou melhores resultados. Algo muito interessante que li também mas para já apenas teorico é inves
de ver o ecossistema como apenas one-way (EC -> LLM), ver isto como uma relação de simbiose/co-evolução. Ou seja, não só usar EC para optimizar LLM, mas também aplicar o contrário.
Seria isto algo interessante para explorar mais no nosso caso?
- Criei também um github caso queiram pedir acesso

## Duvidas/Perguntas
- O que eu procuro exatamente nos artigos? Por exemplo, leio um artigo em que eles tiveram sucesso a otimizar uma LLM evoluindo os pesos... o que faço com isso? É suposto eu pesquisar coisas que poderia implementar no meu trabalho, ou só ter uma ideia do que foi feito?
- Não tenho experiencias em teses, é suposto eu começar a escrever algo, mesmo não sabendo bem que direção irei tomar?
- Começo a ter dificuldade em encontrar artigos relacionados com o que quero fazer... talvez seja melhor agora começar a tentar algo mais prático? Ganhar alguma experiencia talvez?
- Em relação à reunião, finalmente já sei o meu horario definitivo (espero eu) e não tenho aulas terça. Para mim talvez fosse mais conveniente as reuniões serem online, contudo 
estou mais que disposto a continuarem serem presenciais para haver no minimo algum contacto pessoal.

# Reunião 28/10/2025
## TODO!
### Organizar excel (relevance, Type of EC, ...), dropdowns
### VER MAIS KAN, MAMBA
###  Usar nas não so para optimizar, mas também em que a propria arquitetura está em questão !!
###  Fazer uma sopa de arquiteturas (Kan, Mamba, Transformers).
###  NAS para mamba existe? ou para kan? 
### Task: Sumarization text
### Ideia inicial: sopa de arquiteturas e ir evoluindo tudo

Reunião 28/10:
1. Scoope corrigido: Pesquisei apenas algoritmos evolutivos de optimização só para transformers. O objetivo não é esse. O objetivo é que até a própria arquitetura esteja em questão na evolução. Ou seja, queremos que a população inicial seja composta tanto como transformers como KANs e Mambas. Depois sim, utilizar algoritmos evolutivos como o NAS para evoluir as arquiteturas.

2. Todo: 
  - Re-organizar excel de modo a que fique mais filtrável por relevancia, etc
  - Ver/procurar o que existe de optimização por EC para o Mamba e o KAN
  - Incorporar alguma pesquisa mais prática de modo a entender melhor as diferentes arquiteturas e, para caso não exista nada sobre EC, explorar tecnicas que dariam para implementar no Mamba e no KAN