# Atividades em progresso:

- [ ] Implementar processamento em múltiplas GPUs usando nn.parallel.DistributedDataParellel ao invés de nn.DataParallel 
  (a documentação do PyTorch diz que o primeiro tem mais desempenho que o segundo);
- [ ] Treinar só o módulo de estimativa de objetivos primeiro. Pegar os pesos, carregar no modelo completo e aí sim usar a multitask loss;
- [ ] RMSProp e Nesterov (vai bem em VAEs segundo Descending through a Crowded Valley — Benchmarking Deep Learning Optimizers);
- [ ] Implementar Stepwise transformer?
- [ ] Implementar control VAE;
- [ ] Implementar FDS ;
- [X] Implementar Balanced MSE loss (https://github.com/jiawei-ren/BalancedMSE);
- [ ] É possível restringir o início da predição da rede? Mesmo prevento o deslocamento e não a posição, muitas vezes a primeira predição já é totalmente
  desconectada da trajetória observada. Consigo forçar que o primeiro deslocamento predito sempre comece em 0 com a inicialização dos pesos? Pesos na loss?

# Lista de ideias

- [ ] Treinar só o módulo de estimativa de objetivos primeiro. Pegar os pesos, carregar no modelo completo e aí sim usar a multitask loss;
- [ ] Artigo novo da Angelica sobre balancear os gradientes de um treinamento multitarefas considerando os gradientes passados;
- [ ] Investigar saídas e gts da rede. Por que as saídas e os gts são da forma (16, 15, 45, 4) e os gts não são iguais ao longo da dimensão 1? Não deveria
  importar e ser usado nas losses apenas os resultados para o último instante observado? As métricas são calculadas apenas para o último;
- [ ] Adicionar função de ativação no regressor de saída ajudaria na predição já que os valores estão no intervalo [-1, 1]? Tanh, hardtanh ou softsign
  (para o PIE e o JAAD o código usa a tanh, mas para o outro conjunto, não usa nenhuma);
- [ ] Investigar quais as melhores métricas e losses para a otimização;
- [ ] É possível restringir o início da predição da rede? Mesmo prevento o deslocamento e não a posição, muitas vezes a primeira predição já é totalmente
  desconectada da trajetória observada. Consigo forçar que o primeiro deslocamento predito sempre comece em 0 com a inicialização dos pesos?
- [ ] Adicionar função de ativação no regressor de saída ajudaria na predição já que os valores estão no intervalo [-1, 1]? Tanh, hardtanh ou softsign;
- [ ] Investigar efeitos do formato adotado: cxcywh normalizado 0-1 (padrão) ou xyxy;
- [X] Como balancear os três termos da loss function: os três termos têm ordem de grandeza parecida, mas a KLD cai rápido demais e ficam as outras duas,
  sendo que a CVAE loss (BoM MSE) cai mais rápido que a loss de objetivos. Perguntar pro Nicão e para o Valdir. Isso parece um problema de multitask learning,
  mesmo que não sejam aprendidas duas tarefas;
- [ ] Investigar efeito do número de objetivos intermediários na velocidade de predição (segundo o artigo, afeta o desempenho também, mas mesmo números baixos
  já correspondiam ao estado da arte);
- [ ] Early stopping começa a valer só após o fim do período de annealing;
- [ ] Adicionar filtro de Kalman na saída do modelo para tratar a incerteza?
- [ ] Pensar sobre maneiras de ponderar o erro em x e em y na imagem, já que a partir da perspectiva do carro, esses erros têm ordens de grandezas diferentes;
- [ ] KLD não deveria subtrair na loss? Pq está somando?
- [X] Testar se não devo usar a multitask loss somente para os objetivos e para a predição final, deixando o KLD apenas somando na fórmula 
  (essa é a minha interpretação do artigo SceneCode: Monocular Dense Semantic Reconstruction using Learned Encoded Scene Representations);

# Lista de atividades a serem feitas

- [x] Criar repositório no meu GitHub com um fork do repositório original e subir mudanças numa branch separada;
- [x] Revisar modelo e acelerar inferência;
- [x] Revisar métricas calculadas e garantir que elas estão de acordo com a realidade do PIE;
- [x] ~~Implementar ```eval_jaad_pie_cvae()``` usando ````torch```` e não ````numpy````~~ (por algum motivo ficou mais lento);
- [x] Checar nos scripts ```train_cvae.py``` e ```eval_cvae.py``` se ler os checkpoints está funcionando. Já corrigi o problema
  para o ```test_plot.py```. Preciso ler os pesos antes de chamar nn.DataParallel;
- [X] Implementar wrapper para o modelo conter a loss function e com isso generalizar para os casos multi task ou normal; 
- [X] Implementar parâmetro para ponderar a KL loss e evitar o KL annealing (https://github.com/umautobots/bidireaction-trajectory-prediction/issues/4);
- [X] Implementar FIOU;
- [ ] Implementar processamento em múltiplas GPUs usando nn.parallel.DistributedDataParellel ao invés de nn.DataParallel 
  (a documentação do PyTorch diz que o primeiro tem mais desempenho que o segundo);
- [ ] Medir fps da predição (atualmente o código não considera batch size 1);
- [X] Adicionar imagens com as predições e o ground truth no tensorboard;
- [X] Adicionar telinha do Nicão com matplotlib/opencv que mostra as imagens durante o treinamento;
- [X] Implementar early stopping;
- [X] Arrumar hparams no tensorboard;
- [X] Adicionar AdaBelief;
- [X] Implementar salvamento de imagens de teste;
- [X] Implementar salvamento de imagens de teste mostrando as 20 trajetórias possíveis;
- [X] Implementar LDS (https://arxiv.org/pdf/2102.09554.pdf);
- [ ] Implementar FDS ;
- [X] Implementar Balanced MSE loss (https://github.com/jiawei-ren/BalancedMSE);
- [ ] Estimativa de profundidade usando o nuScenes;
- [ ] Segmentação semântica usando o nuScenes;
- [ ] Odometria;
- [ ] Montar dataset de trajetórias extraídas do nuScenes;
- [ ] Adicionar entradas no modelo (testar diferentes pontos e mecanismos de atenção).

# Informações importantes para serem lembradas:

1) Divisão do PIE (````_get_image_set_ids()````): Mesma divisão adotada pelo PIETraj e pelo BiTraP.

`````        
image_set_nums = {'train': ['set01', 'set02', 'set04'],
                          'val': ['set05', 'set06'],
                          'test': ['set03'],
                          'all': ['set01', 'set02', 'set03',
                                  'set04', 'set05', 'set06']}

``````

2) Reescrevi o ````SummaryWriter()```` do PyTorch:

- Adicionei a flag ``timed`` no método ``add_hparam()``;
- Só no método ``add_hparam()`` a classe cria um objeto igual a ela mesma e esse objeto recebe a flag ``timed``. Se for ``False``, no construtor
  eu faço ``self._get_file_writer(timed=timed)``, que causa o seguinte:

```
        if self.all_writers is None or self.file_writer is None:
            self.file_writer = FileWriter(self.log_dir, self.max_queue,
                                          self.flush_secs, self.filename_suffix, timed=timed)

```

- Na classe ``FileWriter``, crio um objeto ``EventFileWriter`` (```event_file_writer.py```) que recebe a flag. Se for falsa, não uso
  nem o relógio e nem o global uid no construtor para definir o nome do arquivo. 
  
Fim :)))))))))))