# ==============================================================================
# PACOTES
# ==============================================================================

pacotes <- c(
  "caret","dplyr","smotefamily","pROC",
  "doParallel","ggplot2","ranger","gbm",
  "FSelector","e1071","naivebayes","nnet",
  "factoextra","FactoMineR"
)

for(p in pacotes){
  if(!require(p, character.only = TRUE)){
    install.packages(p)
    library(p, character.only = TRUE)
  }
}

# ==============================================================================
# BASE
# ==============================================================================

url <- "C:/Users/Cristian/Downloads/Código artigo final - data mining/Código artigo final - data mining/dados/Indian Liver Patient Dataset (ILPD).csv"

col_names <- c("Age","Gender","TB","DB","Alkphos","Sgpt",
               "Sgot","TP","ALB","AG_Ratio","Class")

bd <- read.csv(url, header = FALSE, col.names = col_names)

bd$Gender <- ifelse(bd$Gender == "Female", 0, 1)
bd$AG_Ratio[is.na(bd$AG_Ratio)] <- mean(bd$AG_Ratio, na.rm = TRUE)

bd$Class <- factor(bd$Class, levels = c(1,2), labels = c("no","yes"))

# ==============================================================================
# PCA (VISUAL)
# ==============================================================================

pca_res <- prcomp(bd[, -ncol(bd)], scale. = TRUE)

fviz_pca_ind(
  pca_res,
  geom.ind = "point",
  col.ind = bd$Class,
  palette = c("#00AFBB", "#FC4E07"),
  addEllipses = TRUE
)

# ==============================================================================
# CONFIG
# ==============================================================================

set.seed(123)
N_REP <- 10

fitControl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "smote"
)

# ==============================================================================
# MODELOS
# ==============================================================================

metodos <- list(
  "Logistic" = "glm",
  "Random Forest" = "ranger",
  "SVM" = "svmRadial",
  "GBM" = "gbm",
  "Naive Bayes" = "naive_bayes",
  "ANN" = "nnet"
)

# ==============================================================================
# GRIDS
# ==============================================================================

grid_svm <- expand.grid(
  sigma = c(0.01, 0.05),
  C = c(1, 5, 10)
)

grid_ann <- expand.grid(
  size = c(3,5,7),
  decay = c(0.0, 0.01)
)

grid_gbm <- expand.grid(
  interaction.depth = c(1,3),
  n.trees = c(100,300),
  shrinkage = c(0.05),
  n.minobsinnode = 10
)

# ==============================================================================
# PARALELISMO
# ==============================================================================

cl <- makeCluster(parallel::detectCores()-1)
registerDoParallel(cl)

# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

rodar_modelos <- function(treino, teste, nome_base){
  
  resultados <- data.frame()
  
  for(nome in names(metodos)){
    
    cat("Treinando", nome, "-", nome_base, "\n")
    
    modelo <- try({
      
      if(nome == "SVM"){
        train(Class ~ ., data = treino,
              method = "svmRadial",
              metric = "ROC",
              trControl = fitControl,
              preProcess = c("center","scale","YeoJohnson"),
              tuneGrid = grid_svm)
        
      } else if(nome == "ANN"){
        train(Class ~ ., data = treino,
              method = "nnet",
              metric = "ROC",
              trControl = fitControl,
              preProcess = c("center","scale"),
              tuneGrid = grid_ann,
              trace = FALSE)
        
      } else if(nome == "GBM"){
        train(Class ~ ., data = treino,
              method = "gbm",
              metric = "ROC",
              trControl = fitControl,
              preProcess = c("center","scale"),
              tuneGrid = grid_gbm,
              verbose = FALSE)
        
      } else if(nome == "Random Forest"){
        train(Class ~ ., data = treino,
              method = "ranger",
              metric = "ROC",
              trControl = fitControl,
              num.trees = 1000)
        
      } else {
        train(Class ~ ., data = treino,
              method = metodos[[nome]],
              metric = "ROC",
              trControl = fitControl,
              preProcess = c("center","scale"))
      }
      
    }, silent = TRUE)
    
    if(inherits(modelo, "try-error")) next
    
    prob <- predict(modelo, teste, type = "prob")[,"yes"]
    
    pred <- ifelse(prob > 0.4, "yes", "no")
    pred <- factor(pred, levels = c("no","yes"))
    
    cm <- confusionMatrix(pred, teste$Class)
    
    roc_obj <- roc(teste$Class, prob, levels = rev(levels(teste$Class)))
    
    # balanced accuracy
    balanced <- ((cm$byClass["Sensitivity"] + cm$byClass["Specificity"]) / 2) * 100
    
    resultados <- rbind(resultados, data.frame(
      Base = nome_base,
      Metodo = nome,
      Accuracy = round(cm$overall["Accuracy"]*100,2),
      Sensitivity = round(cm$byClass["Sensitivity"]*100,2),
      Specificity = round(cm$byClass["Specificity"]*100,2),
      Balanced = round(balanced,2),
      AUC = round(auc(roc_obj)*100,2)
    ))
  }
  
  return(resultados)
}

# ==============================================================================
# LOOP PRINCIPAL
# ==============================================================================

resultados_finais <- data.frame()

for(i in 1:N_REP){
  
  cat("\n========== REPETIÇÃO", i, "==========\n")
  
  idx <- createDataPartition(bd$Class, p = 0.7, list = FALSE)
  
  treino <- bd[idx, ]
  teste  <- bd[-idx, ]
  
  # CFS antes
  vars_cfs <- cfs(Class ~ ., treino)
  
  treino_cfs <- treino[, c(vars_cfs,"Class")]
  teste_cfs  <- teste[, c(vars_cfs,"Class")]
  
  # SMOTE apenas para seleção
  sm <- SMOTE(treino[, -ncol(treino)], treino$Class)
  treino_sm <- sm$data
  colnames(treino_sm)[ncol(treino_sm)] <- "Class"
  
  # CFS depois
  vars_cfs2 <- cfs(Class ~ ., treino_sm)
  
  treino_cfs2 <- treino_sm[, c(vars_cfs2,"Class")]
  teste_cfs2  <- teste[, c(vars_cfs2,"Class")]
  
  resultados_finais <- rbind(
    resultados_finais,
    rodar_modelos(treino, teste, "Original"),
    rodar_modelos(treino_cfs, teste_cfs, "CFS Antes"),
    rodar_modelos(treino_cfs2, teste_cfs2, "CFS Depois")
  )
}

stopCluster(cl)

# ==============================================================================
# RESULTADO FINAL
# ==============================================================================

media_final <- resultados_finais %>%
  group_by(Base, Metodo) %>%
  summarise(
    Acc = round(mean(Accuracy),2),
    Sen = round(mean(Sensitivity),2),
    Spe = round(mean(Specificity),2),
    Bal = round(mean(Balanced),2),
    AUC = round(mean(AUC),2)
  )

print(media_final)

# ==============================================================================
# GRÁFICO
# ==============================================================================

ggplot(media_final, aes(x = Metodo, y = AUC, fill = Base)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Comparação Final (AUC)")

# ==============================================================================
# PCA PÓS SMOTE (APENAS VISUAL)
# ==============================================================================

X <- treino[, -ncol(treino)]
y <- treino$Class

smote_data <- SMOTE(X, y, K = 5)

treino_smote <- smote_data$data
colnames(treino_smote)[ncol(treino_smote)] <- "Class"

treino_smote$Class <- as.factor(treino_smote$Class)

res.pca.smote <- PCA(treino_smote[,-ncol(treino_smote)], graph = FALSE)

fviz_pca_ind(
  res.pca.smote,
  geom.ind = "point",
  col.ind = treino_smote$Class,
  palette = c("#00AFBB", "#E7B800"),
  addEllipses = TRUE,
  legend.title = "Groups",
  title = "PCA - Pós SMOTE"
)

table(treino_smote$Class)