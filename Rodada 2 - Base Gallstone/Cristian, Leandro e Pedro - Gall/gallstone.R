# ==============================================================================
# INSTALAÇÃO DE PACOTES
# ==============================================================================

if(!require(caret)) install.packages("caret")
if(!require(dplyr)) install.packages("dplyr")
if(!require(randomForest)) install.packages("randomForest")
if(!require(e1071)) install.packages("e1071")
if(!require(naivebayes)) install.packages("naivebayes")
if(!require(FSelector)) install.packages("FSelector")
if(!require(nnet)) install.packages("nnet")
if(!require(pROC)) install.packages("pROC")
if(!require(readxl)) install.packages("readxl")
if(!require(doParallel)) install.packages("doParallel")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(reshape2)) install.packages("reshape2")
if(!require(smotefamily)) install.packages("smotefamily")
if(!require(gbm)) install.packages("gbm")

# ==============================================================================
# PACOTES
# ==============================================================================

library(caret)
library(dplyr)
library(randomForest)
library(e1071)
library(naivebayes)
library(FSelector)
library(nnet)
library(pROC)
library(readxl)
library(doParallel)
library(ggplot2)
library(reshape2)
library(smotefamily)
library(gbm)

# ==============================================================================
# CARREGAMENTO DA BASE (LOCAL)
# ==============================================================================

url <- "D:/projetos/R-Studio/Base2/Dados/basegal.xlsx"

# Se for Excel:
bd <- read_excel(url)



bd <- as.data.frame(bd)

colnames(bd) <- make.names(colnames(bd))

# ==============================================================================
# PRÉ-PROCESSAMENTO
# ==============================================================================

colnames(bd)
# Detectar automaticamente a variável alvo
col_classe <- grep("gall|class|status", colnames(bd), ignore.case = TRUE, value = TRUE)

bd$Class <- bd[[col_classe[1]]]

# Converter para fator
bd$Class <- factor(bd$Class)

# Padronizar nomes das classes
levels(bd$Class) <- c("NoDisease","Disease")

bd <- bd[, !(colnames(bd) %in% col_classe)]

bd <- bd %>%
  mutate(across(where(is.character), as.factor))

bd <- na.omit(bd)

table(bd$Class)

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

fitControl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE,
  search = "random"    # Explora melhor os hiperparâmetros
)

metodos <- list(
  "Logistic" = "glm",
  "Random Forest" = "ranger",
  "SVM" = "svmRadial",
  "Naive Bayes" = "naive_bayes",
  "ANN" = "nnet",
  "GBM" = "gbm"
)

grid_svm <- expand.grid(
  C = 2^(-2:4),
  sigma = 2^(-8:-1)
)

# ==============================================================================
# FUNÇÃO DE TREINAMENTO
# ==============================================================================
set.seed(123)
rodar_modelos <- function(base, nome_cenario){
  
  set.seed(123)
  idx <- createDataPartition(base$Class, p = 0.80, list = FALSE)
  
  treino <- base[idx,, drop = FALSE]
  teste  <- base[-idx,, drop = FALSE]
  
  resultados <- data.frame()
  roc_list <- list()
  
  for(nome in names(metodos)){
    
    cat("\n---", nome_cenario, "-", nome, "---\n")
    
    # Parâmetros base
    args <- list(
      form = Class ~ .,
      data = treino,
      method = metodos[[nome]],
      trControl = fitControl,
      metric = "ROC",
      preProcess = c("center","scale")
    )
    
    # Ajustes específicos por modelo
    if(metodos[[nome]] == "svmRadial"){
      args$tuneGrid <- grid_svm 
    } else if(metodos[[nome]] == "nnet"){
      args$tuneLength <- 15 # Aumentado pois o random search explorará mais combinações
      args$trace <- FALSE
      args$MaxNWts <- 5000
    } else {
      args$tuneLength <- 15 
    }
    
    # Treinamento
    modelo <- do.call(train, args)
    
    # Predições
    pred <- predict(modelo, teste)
    prob <- predict(modelo, teste, type = "prob")
    
    # Métricas
    cm <- confusionMatrix(pred, teste$Class)
    roc_obj <- roc(teste$Class, prob$Disease, levels = rev(levels(teste$Class)))
    
    # imp <- varImp(modelo)
    # print(plot(imp, main = paste("Importância:", nome)))
    
    resultados <- rbind(resultados, data.frame(
      Cenario = nome_cenario,
      Metodo = nome,
      Accuracy = round(cm$overall["Accuracy"]*100,2),
      Sensitivity = round(cm$byClass["Sensitivity"]*100,2),
      Specificity = round(cm$byClass["Specificity"]*100,2),
      AUC = round(auc(roc_obj)*100,2)
    ))
    
    roc_list[[paste(nome_cenario, nome, sep = " - ")]] <- roc_obj
  }
  
  return(list(resultados = resultados, roc = roc_list))
}

# ==============================================================================
# PARALELISMO
# ==============================================================================

cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# ==============================================================================
# CENÁRIO 1 - TODAS VARIÁVEIS
# ==============================================================================
set.seed(123)
res_all <- rodar_modelos(bd, "Todas Variáveis")

# ==============================================================================
# TESTE PROGRESSIVO DE VARIÁVEIS (1 A 7)
# ==============================================================================

weights <- information.gain(Class ~ ., data = bd)
n_features <- 7

features_names <- cutoff.k(weights, n_features)
# Objeto para acumular todos os resultados progressivos
resultados_progressivos <- data.frame()
lista_roc_progressiva <- list()


# Loop de 1 até o número total de variáveis selecionadas (7)
for(i in 1:n_features) {
  
  # Seleciona as 'i' primeiras variáveis da lista ordenada
  vars_da_vez <- features_names[1:i]
  
  cat("\n======================================================\n")
  cat("Testando Cenário com as Top", i, "Variáveis\n")
  cat("Variáveis incluídas:", paste(vars_da_vez, collapse = ", "), "\n")
  cat("======================================================\n")
  
  # Criar o subconjunto da base com as 'i' variáveis + a classe
  bd_temp <- bd[, c(vars_da_vez, "Class"), drop = FALSE]
  
  # Executar a sua função rodar_modelos
  nome_cenario <- paste("Top", i, "Vars")
  res_temp <- rodar_modelos(bd_temp, nome_cenario)
  
  # Acumular os resultados na tabela final
  resultados_progressivos <- rbind(resultados_progressivos, res_temp$resultados)
  
  # Guardar as curvas ROC caso queira plotar depois
  lista_roc_progressiva[[nome_cenario]] <- res_temp$roc
}

# ==============================================================================
# VISUALIZAÇÃO DOS RESULTADOS PROGRESSIVOS
# ==============================================================================

# Exibir a tabela final com o comparativo de todos os passos
print(resultados_progressivos)

# Gráfico para ver a evolução da Acurácia conforme aumentamos as variáveis
ggplot(resultados_progressivos, aes(x = factor(Cenario, levels = unique(Cenario)), 
                                    y = Accuracy, 
                                    group = Metodo, 
                                    color = Metodo)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(
    title = "Evolução da Acurácia por Número de Variáveis",
    x = "Cenário (Quantidade de Variáveis)",
    y = "Acurácia (%)",
    color = "Algoritmo"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ==============================================================================
# CONSOLIDAÇÃO DOS RESULTADOS
# ==============================================================================

# Unir o cenário de "Todas Variáveis" com os testes progressivos
resultados_finais <- rbind(res_all$resultados, resultados_progressivos)

# Garantir que a ordem dos cenários no gráfico siga a lógica (Todas -> Top 1 -> Top 7)
resultados_finais$Cenario <- factor(resultados_finais$Cenario, 
                                    levels = unique(resultados_finais$Cenario))

# ==============================================================================
# COMPARAÇÃO MÉDIA POR CENÁRIO
# ==============================================================================

cat("\n==============================\n")
cat("Resumo de Performance por Cenário (Média dos Modelos)\n")
cat("==============================\n")

comparacao <- resultados_finais %>%
  group_by(Cenario) %>%
  summarise(
    Accuracy_media    = round(mean(Accuracy), 2),
    Sensitivity_media = round(mean(Sensitivity), 2),
    Specificity_media = round(mean(Specificity), 2),
    AUC_media         = round(mean(AUC), 2)
  ) %>%
  arrange(desc(AUC_media)) # Ordena pelos melhores resultados de AUC

print(comparacao)

# ==============================================================================
# VISUALIZAÇÃO FACETADA (MÉTRICAS POR MODELO E CENÁRIO)
# ==============================================================================


resultados_long <- melt(resultados_finais, id.vars = c("Metodo", "Cenario"))

ggplot(resultados_long, aes(x = Metodo, y = value, fill = Cenario)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~variable, scales = "free_y") +
  theme_minimal() +
  scale_fill_viridis_d(option = "mako") + # Cores melhores para muitos cenários
  labs(
    title = "Comparação Detalhada: Todas Variáveis vs. Seleção Progressiva",
    subtitle = "Métricas de desempenho por algoritmo e subconjunto de dados",
    y = "Valor (%)",
    fill = "Cenário"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ==============================================================================
# CONSOLIDAÇÃO DAS CURVAS ROC
# ==============================================================================

# A lista_roc_progressiva é uma lista de listas (cenário -> modelos)
# Vamos extrair apenas o melhor modelo de cada cenário para não poluir o gráfico
# ou unir todos se preferir:

roc_total <- res_all$roc # Começa com as de "Todas Variáveis"

for(cenario in names(lista_roc_progressiva)){
  roc_total <- c(roc_total, lista_roc_progressiva[[cenario]])
}

cat("\nTotal de curvas ROC geradas:", length(roc_total), "\n")

# ==============================================================================
# GRÁFICO DE ACURÁCIA
# ==============================================================================

ggplot(resultados_finais, aes(x = Metodo, y = Accuracy, fill = Cenario)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Comparação de Acurácia",
       y = "Acurácia (%)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ==============================================================================
# ROC GERAL
# ==============================================================================

plot(roc_total[[1]], col = 1, main = "Curvas ROC", lwd = 2)

i <- 2
for(nome in names(roc_total)){
  plot(roc_total[[nome]], col = i, add = TRUE, lwd = 2)
  i <- i + 1
}

legend("bottomright",
       legend = names(roc_total),
       col = 1:length(roc_total),
       lwd = 2,
       cex = 0.6)

stopCluster(cl)

# ==============================================================================
# RANKING DE IMPORTÂNCIA DAS VARIÁVEIS (INFORMATION GAIN)
# ==============================================================================

# 1. Transformar os pesos em um data frame amigável para o ggplot
df_weights <- data.frame(
  Variavel = rownames(weights),
  Importancia = weights$attr_importance
)

# 2. Ordenar o data frame para o gráfico ficar em ordem decrescente
df_weights <- df_weights %>%
  arrange(desc(Importancia))

# 3. Criar o gráfico de ranking
ggplot(df_weights, aes(x = reorder(Variavel, Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(Importancia, 3)), hjust = -0.2, size = 3) + # Adiciona os valores nas barras
  coord_flip() + # Inverte para facilitar a leitura dos nomes das variáveis
  theme_minimal() +
  labs(
    title = "Ranking de Importância das Variáveis",
    subtitle = "Baseado em Information Gain",
    x = "Variáveis",
    y = "Ganho de Informação"
  ) +
  # Ajusta o limite do eixo Y para o texto não ser cortado
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) 

# Exibir a tabela no console também, se desejar
print(df_weights)