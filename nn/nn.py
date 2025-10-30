import numpy        as np
import numpy.typing as npt

class NeuralNetwork:
    training_input:     npt.NDArray[np.float32]
    training_output:    npt.NDArray[np.float32]
    enter_weights:      npt.NDArray[np.float32]
    inner_weights:      npt.NDArray[np.float32]
    exit_weights:       npt.NDArray[np.float32]

    _inner_outputs:     npt.NDArray[np.float32]
    _inner_errors:      npt.NDArray[np.float32]
    
    _output:            np.float32
    _error:             np.float32

    nodes_per_layer:    int
    layer_count:        int
    attribute_count:    int
    training_dat_count: int

    def NeuralNetwork(
            self,
            attribute_count: int,
            layer_count: int, 
            perceptron_per_layer_count: int,
            training_dat_count: int
            ):

        self.nodes_per_layer    = perceptron_per_layer_count
        self.layer_count        = layer_count
        self.attribute_count    = attribute_count
        self.training_dat_count = training_dat_count

        self.training_input = np.empty(
                shape=(training_dat_count, attribute_count),
                dtype=np.float32
                )
        self.training_ouput = np.empty(
                shape=(training_dat_count,),
                dtype=np.float32
                )

        # enterWeights[i, j] := weight from enter layer node i
        # to 0th inner layer node j
        self.enter_weights = np.zeros(
                shape=(attribute_count + 1,
                       self.nodes_per_layer),
                dtype=np.float32
                )

        # inner_weights[i, j, k] := weight from inner layer i node j
        # to inner layer i+1 node k
        self.inner_weights = np.zeros(
                shape=(self.layer_count - 1,
                       self.nodes_per_layer + 1,
                       self.nodes_per_layer),
                dtype=np.float32
                )

        # exit_weights[i] := weight from inner layer n-1 node i
        # to result layer node
        self.exit_weights = np.zeros(
                shape=(self.nodes_per_layer + 1,),
                dtype=np.float32
                )

    def forward(self, x: npt.NDArray[np.float32]):
        pass
