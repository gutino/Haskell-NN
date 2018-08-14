import Data.List

--Recebe o input da primeira layer [Float] e a rede (os pesos), calcula a soma
-- i*w e aplica a funcao de ativacao para cada neuronio. Devolve uma lista de listas,
--onde cada lista eh o output para cada layer

--Parte do Gustavo

type Network = [[[Float]]]

makeList :: Int -> [Float]
makeList 0 = [1.0]
makeList x = 1.0 : (makeList (x-1))

makeMatrix :: Int -> Int -> [[Float]]
makeMatrix 0 y = []
makeMatrix x y = ((makeList y) : (makeMatrix (x-1) y))

--Recebe uma lista com o numero de neuronios por layer e devolve uma lista de matrizes
--com os pesos das ligacoes
createNetwork :: [Int] -> Network
createNetwork [x]       = []
createNetwork (x:y:ys) = (makeMatrix y x) : (createNetwork (y:ys))

--Parte do Thiago

sigmoid :: Float -> Float -> Float
sigmoid x a = 1/(1 + exp(-a*x))

ativacao :: Float -> Float
ativacao a | a < 2 = 0
           | otherwise = 1

-- Dados vetor de entradas, matriz de pesos e função de ativação
-- retorna a lista de output da camada
calculaAct :: [Float] -> [[Float]] -> (Float->Float) -> [Float]
calculaAct xs [] f      = []
calculaAct xs (y:ys) f  = [f $ sum $ zipWith (*) xs y] ++ calculaAct xs ys f

-- Dados vetor de entradas, rede neural e função de ativação
-- retorna a lista de output e pesos da camada
calcOutput :: [Float] -> Network -> (Float -> Float) -> [[Float]]
calcOutput xs [] f     = []
calcOutput xs (r:rs) f = [resultCamada] ++ calcOutput resultCamada rs f
    where
        resultCamada = calculaAct (xs++[-1.0]) r f

-- createNetwork [1,2,3]
-- -->[[[1.0,1.0],[1.0,1.0]],[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]]
-- cria rede com 1 entrada + bias, 2 neurônios de entrada + bias e 3 neurônios de saída + bias

-- calcOutput [1] (createNetwork [1,2,3]) (`sigmoid` 3.0)
-- -->[[0.5,0.5],[0.5,0.5,0.5]]
-- A rede acima é criada e a entrada 1 é aplicada à ela. É criada automaticamente uma entrada fixa -1 para cada neuonio,
-- representando a plasticidade intrínseca

--Fim da Parte do Thiago

--derivada da sigmoide
sigmoide' :: Float -> Float
sigmoide' x = (exp (-x)) / ((1 + exp (-x))**2)

--Calcula o gradiente de um neuron da camada de output
-- g = (target - output)*sigmoide'(soma dos neuronios)
outputNeuronGradient :: Float -> Float -> Float -> Float
outputNeuronGradient target output neuronSum = (target-output) * sigmoide' neuronSum

--Calcula o gradiente de todos os neurons da camada de output (um por vez)
outputLayerGradients :: [Float] -> [Float] -> [Float] -> [Float]
outputLayerGradients targets output sums = [outputNeuronGradient t o s | (t, o, s) <- zip3 targets output sums]

--Calcula o gradiente de um neuron das camadas intermediarias
--g = sum (w*g) * sigmoide' (sumNeuron)
neuronGradient :: [Float] -> [Float] -> Float -> Float
neuronGradient weights prevGradients output = (sum [w*g | (w, g) <- zip weights prevGradients]) * sigmoide' output

--Calcula o gradiente de todos os neurons das camadas intermediarias
layerGradients :: [[Float]] -> [Float] -> [Float] -> [Float]
layerGradients weights prevGradients sums = [neuronGradient w prevGradients s | (w, s) <- zip neurons sums]
                                          where neurons = [[neuron !! n | neuron <- weights] | n <- [0..length (head weights)-1]]

--Calcula o gradiente de todas as camadas, comecando pela de output e indo no sentido contrario
networkGradients :: Network -> [[Float]] -> [Float] -> [[Float]]
networkGradients net sums targets = (calcPerLayer net (init sums) outputGradients) ++ [outputGradients]
                                  where outputGradients = outputLayerGradients targets (map sigmoide' outputSum) outputSum
                                        outputSum = last sums
                                        calcPerLayer _ [] _ = []
                                        calcPerLayer net sums lastGradients =  (calcPerLayer (init net) (init sums) layerGradient) ++ [layerGradient]
                                                                              where layerGradient = layerGradients (last net) lastGradients (last sums)

--Recebe a rede, o gamma os gradientes ajusta os pesos:
--Calcula a variacao
--Ajusta os pesos
ajustaPesos :: Network -> Float -> [[Float]] -> Network
ajustaPesos net gamma grad = [[zipWith (+) x y | (x, y) <- zip xs ys] | (xs, ys) <- zip net toFloatList]
                           where
                             netlength = map (map (length)) net
                             toFloatList = [[makeLine x y | (x,y) <- zip xs ys]|(xs, ys) <- zip netlength term]
                             term = map (map (* (negate gamma))) grad
                             makeLine 1 value = [value]
                             makeLine x value = 0.0 : (makeLine (x-1) value)
-- [[(y !! 1) - gama*grad|y <- x]| x <- net]
-- [[[m + n|(m,n) <- zip o p] | (o, p) <- zip q r] | (q, r) <- zip net toFloatList]

--Recebe a rede e um conjunto de treinamento de tuplas (input, output)
--calcula a saida, ajusta os pesos
--Repete até que o criterio de parada (float) seja atingido
--Devolve os pesos finais

-- SUPOSTO TRAINING

training :: Network -> [Float] -> [Float] -> Float -> Float -> Network
training net _ _ _ 0 = net
training net input target gamma step = training (ajustaPesos net gamma gradients) input target gamma (step-1)
                            where
                              outputs   = calcOutput input net (sigmoid 1)
                              deltas    = networkDeltas net outputs target
                              gradients = calcGradients deltas (input:outputs)
