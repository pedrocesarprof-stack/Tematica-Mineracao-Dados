# ==============================================================================
# PACOTES
# ==============================================================================

pacotes <- c(
  "caret","dplyr","readr","smotefamily","pROC",
  "doParallel","ggplot2","reshape2","gbm","ranger",
  "FactoMineR","factoextra","FSelector"
)

for(p in pacotes){
  if(!require(p, character.only = TRUE)){
    install.packages(p)
    library(p, character.only = TRUE)
  }
}

# ==============================================================================
# BASE LOCAL
# ==============================================================================

pasta <- "C:/Users/Cristian/Downloads/drug_induced_autoimmunity_prediction/"
files <- list.files(pasta, full.names = TRUE)

arquivo_treino <- files[grep("train", files, ignore.case = TRUE)][1]
arquivo_teste  <- files[grep("test", files, ignore.case = TRUE)][1]

if(is.na(arquivo_treino)) arquivo_treino <- files[1]
if(is.na(arquivo_teste))  arquivo_teste  <- files[2]

treino <- read.csv(arquivo_treino)
teste  <- read.csv(arquivo_teste)

treino <- as.data.frame(treino)
teste  <- as.data.frame(teste)

# ==============================================================================
# TRATAMENTO INICIAL
# ==============================================================================

remover_cols <- c("ID","SMILES")

treino <- treino[, !(colnames(treino) %in% remover_cols)]
teste  <- teste[, !(colnames(teste) %in% remover_cols)]

colnames(treino)[colnames(treino) == "Label"] <- "Class"
colnames(teste)[colnames(teste) == "Label"]  <- "Class"

treino$Class <- ifelse(treino$Class %in% c(1,"1","yes","positive"), "yes", "no")
teste$Class  <- ifelse(teste$Class  %in% c(1,"1","yes","positive"), "yes", "no")

treino$Class <- factor(treino$Class, levels = c("no","yes"))
teste$Class  <- factor(teste$Class,  levels = c("no","yes"))

# ==============================================================================
# REMOVER VARIÁVEIS PROBLEMÁTICAS
# ==============================================================================

nzv <- nearZeroVar(treino)
if(length(nzv) > 0){
  treino <- treino[, -nzv]
  teste  <- teste[, -nzv]
}

cor_matrix <- cor(treino[, sapply(treino, is.numeric)])
high_corr <- findCorrelation(cor_matrix, cutoff = 0.9)

if(length(high_corr) > 0){
  treino <- treino[, -high_corr]
  teste  <- teste[, -high_corr]
}

# ==============================================================================
# LIMPEZA E DUMMIES
# ==============================================================================

treino <- treino[, colSums(is.na(treino)) == 0]
teste  <- teste[, colnames(treino)]

colnames(treino) <- make.names(colnames(treino))
colnames(teste)  <- make.names(colnames(teste))

dummies <- dummyVars(Class ~ ., data = treino)

treino_x <- predict(dummies, newdata = treino) %>% as.data.frame()
teste_x  <- predict(dummies, newdata = teste) %>% as.data.frame()

common_cols <- intersect(colnames(treino_x), colnames(teste_x))

treino_x <- treino_x[, common_cols, drop = FALSE]
teste_x  <- teste_x[, common_cols, drop = FALSE]

treino <- cbind(treino_x, Class = treino$Class)
teste  <- cbind(teste_x, Class = teste$Class)

# ==============================================================================
# CFS ANTES DO SMOTE
# ==============================================================================

cat("\n================ CFS ANTES DO SMOTE ================\n")

vars_cfs_antes <- cfs(Class ~ ., treino)

print(vars_cfs_antes)

treino_cfs_antes <- treino[, c(vars_cfs_antes, "Class")]
teste_cfs_antes  <- teste[, c(vars_cfs_antes, "Class")]

# ==============================================================================
# SMOTE PARA CFS DEPOIS
# ==============================================================================

X <- treino[, -ncol(treino)]
y <- treino$Class

smote_data <- SMOTE(X, y, K = 5)

treino_smote <- smote_data$data
colnames(treino_smote)[ncol(treino_smote)] <- "Class"
treino_smote$Class <- as.factor(treino_smote$Class)

# ==============================================================================
# CFS DEPOIS DO SMOTE
# ==============================================================================

cat("\n================ CFS DEPOIS DO SMOTE ================\n")

vars_cfs_depois <- cfs(Class ~ ., treino_smote)

print(vars_cfs_depois)

treino_cfs_depois <- treino_smote[, c(vars_cfs_depois, "Class")]
teste_cfs_depois  <- teste[, c(vars_cfs_depois, "Class")]

# ==============================================================================
# PCA (ANTES DO SMOTE)
# ==============================================================================

cat("\n================ PCA RAW ================\n")

res.pca <- PCA(treino[,-ncol(treino)], graph = FALSE)

fviz_pca_ind(res.pca,
             geom.ind = "point",
             col.ind = treino$Class,
             palette = c("#00AFBB", "#E7B800"),
             addEllipses = TRUE,
             legend.title = "Groups",
             title = "RAW"
)

# ==============================================================================
# CONTROLE CARET
# ==============================================================================

fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# SMOTE dinâmico
prop_min_class <- min(prop.table(table(treino$Class)))

if(prop_min_class < 0.4){
  fitControl$sampling <- "smote"
  cat(">> SMOTE ATIVADO NOS MODELOS\n")
}

# ==============================================================================
# MODELOS
# ==============================================================================

metodos <- list(
  "Logistic" = "glm",
  "Random Forest" = "ranger",
  "SVM" = "svmRadial",
  "GBM" = "gbm"
)

cl <- makeCluster(parallel::detectCores()-1)
registerDoParallel(cl)

resultados_finais <- data.frame()

# ==============================================================================
# FUNÇÃO DE TREINO
# ==============================================================================

rodar_modelos <- function(dados_treino, dados_teste, nome_base){
  
  resultados <- data.frame()
  
  for(nome in names(metodos)){
    
    cat("Treinando", nome, "-", nome_base, "\n")
    
    modelo <- train(
      Class ~ .,
      data = dados_treino,
      method = metodos[[nome]],
      metric = "ROC",
      trControl = fitControl,
      preProcess = c("center","scale"),
      tuneLength = 5
    )
    
    prob <- predict(modelo, dados_teste, type = "prob")
    
    if(!"yes" %in% colnames(prob)){
      prob$yes <- prob[,2]
    }
    
    pred <- predict(modelo, dados_teste)
    
    cm <- confusionMatrix(pred, dados_teste$Class)
    
    roc_obj <- roc(
      dados_teste$Class,
      prob[, "yes"],
      levels = rev(levels(dados_teste$Class)),
      quiet = TRUE
    )
    
    resultados <- rbind(resultados, data.frame(
      Base = nome_base,
      Metodo = nome,
      Accuracy = round(cm$overall["Accuracy"]*100,2),
      Sensitivity = round(cm$byClass["Sensitivity"]*100,2),
      Specificity = round(cm$byClass["Specificity"]*100,2),
      AUC = round(auc(roc_obj)*100,2)
    ))
  }
  
  return(resultados)
}

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

resultados_finais <- rbind(
  rodar_modelos(treino, teste, "Original"),
  rodar_modelos(treino_cfs_antes, teste_cfs_antes, "CFS Antes SMOTE"),
  rodar_modelos(treino_cfs_depois, teste_cfs_depois, "CFS Depois SMOTE")
)

stopCluster(cl)

# ==============================================================================
# RESULTADOS
# ==============================================================================

print(resultados_finais)

# ==============================================================================
# GRÁFICO
# ==============================================================================

g <- ggplot(resultados_finais, aes(x = Metodo, y = AUC, fill = Base)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Comparação de AUC")

print(g)

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

# ==============================================================================
# DISTRIBUIÇÕES
# ==============================================================================

cat("\nDistribuição original:\n")
print(table(treino$Class))

cat("\nDistribuição SMOTE:\n")
print(table(treino_smote$Class))
