
# Alunos: Pedro Cesar Rocha, Cristian Martins, Leandro Rosa
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
if(!require(corrplot)) install.packages("corrplot")

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
library(corrplot)

# ==============================================================================
# CARREGAMENTO DA BASE
# ==============================================================================

url <- "C:/Users/Cristian/Desktop/Mineração de dados/Base 1/base1.xls"

bd <- read_excel(url, skip = 1)
bd <- as.data.frame(bd)

colnames(bd)[25] <- "Class"

bd$Class <- factor(
  bd$Class,
  levels = c(0,1),
  labels = c("NoDefault","Default")
)

bd <- bd[,-1]  # remove ID

# ==============================================================================
# ANÁLISE EXPLORATÓRIA
# ==============================================================================

# 1. Distribuição da Classe
ggplot(bd, aes(x = Class, fill = Class)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Distribuição das Classes", y = "Quantidade")

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

prop_treino <- 0.70

fitControl <- trainControl(
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

metodos <- list(
  "Logistic Regression" = "glm",
  "Random Forest" = "ranger",
  "SVM" = "svmRadial",
  "Naive Bayes" = "naive_bayes",
  "ANN" = "nnet"
)

# Grid SVM
C <- 2^c(-1,1,3)
sigma <- 2^c(-5,-3,-1)
grid_svm <- expand.grid(C=C, sigma=sigma)

set.seed(123)

resultados <- data.frame()
roc_list <- list()

# ==============================================================================
# PARTIÇÃO TREINO / TESTE
# ==============================================================================

idx_treino <- createDataPartition(bd$Class, p=prop_treino, list=FALSE)

treino <- bd[idx_treino,]
teste  <- bd[-idx_treino,]

# ==============================================================================
# PARALELISMO
# ==============================================================================

cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# ==============================================================================
# TREINAMENTO
# ==============================================================================

for(nome_metodo in names(metodos)){
  
  cat("\n--- Treinando:", nome_metodo, "---\n")
  
  args_comuns <- list(
    form = Class ~ .,
    data = treino,
    method = metodos[[nome_metodo]],
    trControl = fitControl,
    metric = "ROC",
    preProcess = c("center", "scale")
  )
  
  if(metodos[[nome_metodo]] == "svmRadial"){
    args_comuns$tuneGrid <- grid_svm
    
  } else if(metodos[[nome_metodo]] == "ranger"){
    args_comuns$tuneLength <- 5
    args_comuns$num.threads <- parallel::detectCores() - 1
    
  } else if(metodos[[nome_metodo]] == "nnet"){
    args_comuns$tuneLength <- 5
    args_comuns$trace <- FALSE
    args_comuns$MaxNWts <- 2000
    
  } else {
    args_comuns$tuneLength <- 5
  }
  
  modelo <- do.call(train, args_comuns)
  
  # Predições
  pred <- predict(modelo, teste)
  prob <- predict(modelo, teste, type = "prob")
  
  cm <- confusionMatrix(pred, teste$Class)
  
  # ROC
  roc_obj <- roc(
    response = teste$Class,
    predictor = prob$Default,
    levels = rev(levels(teste$Class))
  )
  
  roc_list[[nome_metodo]] <- roc_obj
  
  # Resultados
  resultados <- rbind(resultados, data.frame(
    Metodo = nome_metodo,
    Accuracy = round(cm$overall["Accuracy"] * 100, 2),
    Sensitivity = round(cm$byClass["Sensitivity"] * 100, 2),
    Specificity = round(cm$byClass["Specificity"] * 100, 2),
    AUC = round(auc(roc_obj) * 100, 2)
  ))
}

stopImplicitCluster()

# ==============================================================================
# RESULTADOS
# ==============================================================================

print(resultados)

# ==============================================================================
# GRÁFICO DE ACURÁCIA
# ==============================================================================

ggplot(resultados, aes(x = Metodo, y = Accuracy, group = 1)) +
  geom_line() +
  geom_point(size = 3) +
  theme_minimal() +
  labs(title = "Acurácia dos Modelos",
       y = "Acurácia (%)",
       x = "Modelo") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ==============================================================================
# COMPARAÇÃO GERAL
# ==============================================================================

resultados_long <- melt(resultados, id.vars = "Metodo")

ggplot(resultados_long, aes(x = Metodo, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Comparação Geral dos Modelos",
       y = "%",
       fill = "Métrica") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ==============================================================================
# CURVAS ROC
# ==============================================================================

plot(roc_list[[1]], col = 1, main = "Curvas ROC")

i <- 2
for(nome in names(roc_list)){
  if(i == 2){
    plot(roc_list[[nome]], col = i, add = TRUE)
  } else {
    plot(roc_list[[nome]], col = i, add = TRUE)
  }
  i <- i + 1
}

legend("bottomright",
       legend = names(roc_list),
       col = 1:length(roc_list),
       lwd = 2)