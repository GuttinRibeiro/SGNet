# Atividades em progresso:

- [ ] Implementar processamento em múltiplas GPUs usando nn.parallel.DistributedDataParellel ao invés de nn.DataParallel 
  (a documentação do PyTorch diz que o primeiro tem mais desempenho que o segundo);
- [ ] Checar nos scripts ```train_cvae.py``` e ```eval_cvae.py``` se ler os checkpoints está funcionando. Já corrigi o problema
  para o ```test_plot.py```. Preciso ler os pesos antes de chamar nn.DataParallel;

# Lista de ideias

- [ ] Investigar quais as melhores métricas e loss para a otimização. Observei, por exemplo, que a FIoU sempre dava 0 (NÃO ESTÁ IMPLEMENTADA!!!!) 
  e essa seria uma métrica importante;
- [ ] Investigar efeitos do formato adotado (bounding box ou ponto central);
- [ ] Como balancear os três termos da loss function: os três termos têm ordem de grandeza parecida, mas a KLD cai rápido demais e ficam as outras duas,
  sendo que a CVAE loss (BoM MSE) cai mais rápido que a loss de objetivos. Perguntar pro Nicão e para o Valdir. Isso parece um problema de multitask learning,
  mesmo que não sejam aprendidas duas tarefas;
- [ ] Investigar feito do número de objetivos intermediários na velocidade de predição (segundo o artigo, afeta o desempenho também, mas mesmo números baixos
  já correspondiam ao estado da arte);
- [ ] Investigar o fato de a SGNet usar o Trajectron++. Isso não está no artigo e no código eu vi referência apenas à BiTraP, que também não está no artigo.
  Ver qual é o melhor encoder e se não é possível importar algum peso pré-treinado;
- [ ] Compensa implementar early stopping?
- [ ] Pelo tensorboard, muitos exemplos têm erro muito baixo, enquanto outros apresentam erros muito altos. É possível salvar o frame dessas sequências, mapeá-las
  e aí aplicar alguma técnica específica de aprendizado? Analisar quais são essas situações e ver se as distorções ocorrem para uma ou várias métricas. Talvez
  seja o caso de usar curriculum learning;
- [ ] Adicionar filtro de Kalman na saída do modelo para tratar a incerteza?

# Lista de atividades a serem feitas

- [x] Criar repositório no meu GitHub com um fork do repositório original e subir mudanças numa branch separada;
- [x] Revisar modelo e acelerar inferência;
- [x] Revisar métricas calculadas e garantir que elas estão de acordo com a realidade do PIE;
- [x] ~~Implementar ```eval_jaad_pie_cvae()``` usando ````torch```` e não ````numpy````~~ (por algum motivo ficou mais lento);
- [ ] Checar nos scripts ```train_cvae.py``` e ```eval_cvae.py``` se ler os checkpoints está funcionando. Já corrigi o problema
  para o ```test_plot.py```. Preciso ler os pesos antes de chamar nn.DataParallel;
- [ ] Implementar FIOU;
- [ ] Implementar processamento em múltiplas GPUs usando nn.parallel.DistributedDataParellel ao invés de nn.DataParallel 
  (a documentação do PyTorch diz que o primeiro tem mais desempenho que o segundo);
- [ ] Medir fps da predição (atualmente o código não considera batch size 1);
- [X] Adicionar imagens com as predições e o ground truth no tensorboard;
- [X] Adicionar telinha do Nicão com matplotlib/opencv que mostra as imagens durante o treinamento;
- [ ] ~~Adicionar página HTML com imagens de teste, conforme repositório da HRNet da NVIDIA disponível em: 
  https://github.com/NVIDIA/semantic-segmentation/blob/main/utils/results_page.py~~; 
- [X] Arrumar hparams no tensorboard;
- [ ] Colocar código no iluvatar;
- [ ] Estimativa de profundidade usando o nuScenes;
- [ ] Segmentação semântica usando o nuScenes;
- [ ] Odometria;
- [ ] Montar dataset de trajetórias extraídas do nuScenes;
- [ ] Adicionar entradas no modelo (testar diferentes pontos e mecanismos de atenção).

# Informações importantes para serem lembradas:

1) Divisão do PIE (````_get_image_set_ids()````):

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