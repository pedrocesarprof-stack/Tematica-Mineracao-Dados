# ==============================================================================
# INSTALAÇÃO AUTOMÁTICA DE PACOTES
# ==============================================================================

pacotes <- c(
  "caret","dplyr","readr","smotefamily","pROC",
  "doParallel","ggplot2","reshape2","gbm","ranger"
)

for(p in pacotes){
  if(!require(p, character.only = TRUE)){
    install.packages(p)
    library(p, character.only = TRUE)
  }
}

# ==============================================================================
# BASE UCI
# ==============================================================================

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"

bd <- read.table(url, sep = ",", header = FALSE)

colnames(bd) <- c(
  "ID","Age","Gender","Education","Country","Ethnicity",
  "Nscore","Escore","Oscore","Ascore","Cscore","Impulsive","SS",
  "Alcohol","Amphet","Amyl","Benzos","Caff","Cannabis","Choc",
  "Coke","Crack","Ecstasy","Heroin","Ketamine","Legalh","LSD",
  "Meth","Mushrooms","Nicotine","Semer","VSA"
)

bd <- as.data.frame(bd)
bd$ID <- NULL

# ==============================================================================
# LISTA DE DROGAS
# ==============================================================================

drogas <- c(
  "Alcohol","Amphet","Amyl","Benzos","Caff","Cannabis","Choc",
  "Coke","Crack","Ecstasy","Heroin","Ketamine","Legalh","LSD",
  "Meth","Mushrooms","Nicotine","Semer","VSA"
)

# ==============================================================================
# CONTROLE BASE
# ==============================================================================

fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
  # O parâmetro 'sampling' será configurado dinamicamente dentro do loop
)

metodos <- list(
  "Logistic" = "glm",
  "Random Forest" = "ranger",
  "SVM" = "svmRadial",
  "GBM" = "gbm"
)

# ==============================================================================
# PARALELISMO
# ==============================================================================

cl <- makeCluster(parallel::detectCores()-1)
registerDoParallel(cl)

# ==============================================================================
# PIPELINE MULTI-DROGA
# ==============================================================================

resultados_finais <- data.frame()

for(drug in drogas){
  
  cat("\n==============================\n")
  cat("Droga:", drug, "\n")
  cat("==============================\n")
  
  base <- bd
  
  # Binarização correta
  base$Class <- ifelse(base[[drug]] %in% c("CL0","CL1"), "no", "yes")
  base$Class <- factor(base$Class, levels = c("no","yes"))
  
  # Remover todas as drogas 
  base <- base[, !(colnames(base) %in% drogas)]
  
  # Split 80/20
  set.seed(123)
  idx <- createDataPartition(base$Class, p = 0.8, list = FALSE)
  
  treino <- base[idx,]
  teste  <- base[-idx,]
  
  # Garantir fatores corretos
  treino$Class <- factor(treino$Class, levels = c("no","yes"))
  teste$Class  <- factor(teste$Class, levels = c("no","yes"))
  
  # Remover colunas constantes do treino
  treino <- treino[, sapply(treino, function(x) length(unique(x)) > 1)]
  
  # ==============================================================================
  # DUMMIES E PREPARAÇÃO DOS DADOS
  # ==============================================================================
  
  # Criar dummies 
  dummies <- dummyVars(~ ., data = treino[, colnames(treino) != "Class"])
  
  treino_x <- predict(dummies, newdata = treino) %>% as.data.frame()
  teste_x  <- predict(dummies, newdata = teste) %>% as.data.frame()
  
  # Tornar os nomes das colunas válidos para o R
  colnames(treino_x) <- make.names(colnames(treino_x))
  colnames(teste_x)  <- make.names(colnames(teste_x))
  
  # Alinhar colunas 
  common_cols <- intersect(colnames(treino_x), colnames(teste_x))
  
  treino_x <- treino_x[, common_cols, drop = FALSE]
  teste_x  <- teste_x[, common_cols, drop = FALSE]
  
  # Reconstruir as bases de treino e teste finais
  treino <- cbind(treino_x, Class = treino$Class)
  teste  <- cbind(teste_x, Class = teste$Class)
  
  # ==============================================================================
  # ATUALIZAR SAMPLING DO CARET DINAMICAMENTE
  # ==============================================================================
  
  min_class_count <- min(table(treino$Class))
  prop_min_class  <- min(prop.table(table(treino$Class)))
  usar_smote      <- FALSE
  
  if(prop_min_class < 0.4) {
    usar_smote <- TRUE
    if(min_class_count > 20) {
      cat(">> Caret configurado para SMOTE\n")
      fitControl$sampling <- "smote"
    } else {
      cat(">> Caret configurado para UpSample \n")
      fitControl$sampling <- "up"
    }
  } else {
    cat(">> Classes balanceadas\n")
    fitControl$sampling <- NULL
  }
  
  # ==============================================================================
  # MODELOS
  # ==============================================================================
  
  for(nome in names(metodos)){
    
    cat("Treinando modelo:", nome, "\n")
    
    # O caret cuida do SMOTE/UpSample de forma segura por baixo dos panos
    modelo <- train(
      Class ~ .,
      data = treino,
      method = metodos[[nome]],
      metric = "ROC",
      trControl = fitControl,
      preProcess = c("center","scale"),
      tuneLength = 5
    )
    
    pred <- predict(modelo, teste)
    prob <- predict(modelo, teste, type = "prob")
    
    # Garantir nomes corretos nas probabilidades
    if(!"yes" %in% colnames(prob)){
      prob$yes <- prob[,2]
    }
    
    cm <- confusionMatrix(pred, teste$Class)
    
    # Calcular ROC
    roc_obj <- roc(
      teste$Class,
      prob[, "yes"],
      levels = rev(levels(teste$Class)),
      quiet = TRUE
    )
    
    # Salvar resultados
    resultados_finais <- rbind(resultados_finais, data.frame(
      Droga = drug,
      Metodo = nome,
      Accuracy = round(cm$overall["Accuracy"]*100,2),
      Sensitivity = round(cm$byClass["Sensitivity"]*100,2),
      Specificity = round(cm$byClass["Specificity"]*100,2),
      AUC = round(auc(roc_obj)*100,2),
      SMOTE = usar_smote
    ))
  }
}

stopCluster(cl)

# ==============================================================================
# RESULTADOS
# ==============================================================================

cat("\n==============================\n")
cat("RESULTADOS FINAIS\n")
cat("==============================\n")
print(resultados_finais)

# Melhor modelo por droga 
melhores <- resultados_finais %>%
  group_by(Droga) %>%
  slice_max(AUC, n = 1) %>% 
  slice(1) # Caso dê empate, pega o primeiro

cat("\n==============================\n")
cat("MELHOR MODELO POR DROGA (AUC)\n")
cat("==============================\n")
print(melhores)

# ==============================================================================
# RESUMO COM DESVIO PADRÃO
# ==============================================================================

resumo_metricas <- resultados_finais %>%
  group_by(Droga, Metodo) %>%
  summarise(
    Accuracy_mean = mean(Accuracy, na.rm = TRUE),
    Accuracy_sd   = sd(Accuracy, na.rm = TRUE),
    Sens_mean     = mean(Sensitivity, na.rm = TRUE),
    Sens_sd       = sd(Sensitivity, na.rm = TRUE),
    AUC_mean      = mean(AUC, na.rm = TRUE),
    AUC_sd        = sd(AUC, na.rm = TRUE),
    .groups = "drop"
  )

# ==============================================================================
# TOP 3 DROGAS
# ==============================================================================

top3 <- resultados_finais %>%
  group_by(Droga) %>%
  summarise(AUC = mean(AUC)) %>%
  arrange(desc(AUC)) %>%
  slice(1:3)

print(top3)

# ==============================================================================
# DIAGNÓSTICO DAS TOP 3
# ==============================================================================

for(drug in top3$Droga){
  
  cat("\n========================\n")
  cat("Diagnóstico:", drug, "\n")
  cat("========================\n")
  
  base <- bd
  
  base$Class <- ifelse(base[[drug]] %in% c("CL0","CL1"), "no", "yes")
  
  prop <- prop.table(table(base$Class))
  
  cat("Distribuição (%):\n")
  print(round(prop * 100,2))
  
  if(min(prop) > 0.3){
    cat("Classes balanceadas → melhora a acurácia\n")
  } else {
    cat("Classes desbalanceadas → acurácia pode estar inflada\n")
  }
}

# ==============================================================================
# GRÁFICOS
# ==============================================================================

# AUC
g_acc <- ggplot(resumo_metricas, aes(x = Droga, y = Accuracy_mean, fill = Metodo)) +
  
  geom_bar(
    stat = "identity",
    position = position_dodge(0.9),
    color = "black"
  ) +
  
  geom_errorbar(
    aes(
      ymin = Accuracy_mean - Accuracy_sd,
      ymax = Accuracy_mean + Accuracy_sd
    ),
    width = 0.2,
    position = position_dodge(0.9)
  ) +
  
  scale_fill_viridis_d() +
  
  theme_minimal() +
  
  labs(
    title = "Acurácia Média por Droga (%)",
    y = "Acurácia (%)",
    x = "Droga"
  ) +
  
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

print(g_acc)

# Sensibilidade
g2 <- ggplot(resultados_finais, aes(x = Droga, y = Sensitivity, fill = Metodo)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Sensibilidade por Droga (%)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Média dos modelos
media_modelos <- resultados_finais %>%
  group_by(Metodo) %>%
  summarise(
    Accuracy = mean(Accuracy, na.rm = TRUE),
    Sensitivity = mean(Sensitivity, na.rm = TRUE),
    Specificity = mean(Specificity, na.rm = TRUE),
    AUC = mean(AUC, na.rm = TRUE)
  )

media_long <- melt(media_modelos, id.vars = "Metodo")

g3 <- ggplot(media_long, aes(x = Metodo, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~variable, scales = "free") +
  theme_minimal() +
  labs(title = "Média Geral dos Modelos (%)")

# Imprimir os gráficos 
print(g2)
print(g3)