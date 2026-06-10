# ======================================================================
# PACOTES
# ======================================================================
pacotes <- c(
  "readxl",
  "dplyr",
  "ggplot2",
  "factoextra",
  "FactoMineR",
  "cluster",
  "corrplot"
)

for(p in pacotes){
  if(!require(p, character.only = TRUE)){
    install.packages(p)
    library(p, character.only = TRUE)
  }
}

# ======================================================================
# BASE
# ======================================================================
# Assumindo a leitura do CSV original fornecido
url <- "C:/Users/Cristian/Downloads/Cream Cheese/cheese.xls"

bd <- read_excel(url)

# ======================================================================
# AGREGAR DADOS POR PRODUTO 
# ======================================================================
# Em vez de cada linha ser uma avaliação isolada, criamos o "perfil médio"
# de cada queijo, eliminando o ruído das opiniões individuais dos provadores.

colnames(bd) <- make.names(colnames(bd))

bd_agrupado <- bd %>%
  group_by(Product.name) %>%
  # Seleciona todas as colunas que começam com N., E., H., ou M. (Atributos sensoriais)
  summarise(across(starts_with(c("N.", "E.", "H.", "M.")), 
                   ~mean(., na.rm = TRUE))) %>%
  ungroup()

# Criar a matriz numérica e passar os nomes dos queijos para o índice (rownames)
dados_sensoriais <- as.data.frame(bd_agrupado[, -1])
rownames(dados_sensoriais) <- bd_agrupado$Product.name

# ======================================================================
# MATRIZ DE CORRELAÇÃO 
# ======================================================================
# Agora vemos as correlações REAIS entre os atributos dos queijos
cor_matrix <- cor(dados_sensoriais)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.cex = 0.6,
  tl.col = "black",
  title = "Correlação dos Atributos Sensoriais",
  mar = c(0,0,1,0)
)

# ======================================================================
# PCA (Análise de Componentes Principais)
# ======================================================================
# O FactoMineR já faz a padronização internamente com scale.unit = TRUE
pca_res <- PCA(dados_sensoriais, scale.unit = TRUE, graph = FALSE)

# 1. O seu agrupamento natural dos queijos
fviz_pca_ind(
  pca_res,
  repel = TRUE, # Evita sobreposição de nomes
  col.ind = "cos2", # Pinta os pontos consoante a qualidade de representação
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  title = "PCA - Mapa dos Queijos (Média Global dos Provadores)"
)

# 2. As variáveis mais importantes
fviz_pca_var(
  pca_res,
  col.var = "contrib",
  gradient.cols = c("blue","yellow","red"),
  repel = TRUE,
  title = "Atributos Sensoriais que Mais Influenciam"
)

# ======================================================================
# CLUSTERIZAÇÃO AVANÇADA: HCPC
# ======================================================================
# Substituí o K-Means pelo HCPC. O HCPC baseia-se na matriz do PCA e 
# calcula automaticamente o número ideal de clusters (centróides).

set.seed(123)
res.hcpc <- HCPC(pca_res, graph = FALSE)

# ======================================================================
# VISUALIZAÇÃO DOS CLUSTERS
# ======================================================================

# 1. Dendrograma 
fviz_dend(res.hcpc, 
          cex = 0.8, palette = "jco", 
          rect = TRUE, rect_fill = TRUE, rect_border = "jco",
          labels_track_height = 0.8,
          main = "Dendrograma das Famílias de Queijos")

# 2. Mapa de Clusters 
fviz_cluster(
  res.hcpc,
  repel = TRUE,
  show.clust.cent = TRUE, # Mostra a "Média" de cada Cluster
  palette = "jco",
  ggtheme = theme_minimal(),
  main = "Clusters dos Queijos"
)

# ======================================================================
# DESCRIÇÃO ESTATÍSTICA DOS CLUSTERS
# ======================================================================


cat("\n====================================================================\n")
cat("O QUE DEFINE CADA CLUSTER? (Estatística de Atributos)\n")
cat("====================================================================\n\n")

print(res.hcpc$desc.var$quanti)