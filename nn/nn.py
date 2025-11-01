import numpy        as np
import numpy.typing as npt

class NeuralNetwork:    
    training_input:     npt.NDArray[np.float32]
    training_output:    npt.NDArray[np.float32]
    enter_weights:      npt.NDArray[np.float32]
    inner_weights:      npt.NDArray[np.float32]
    exit_weights:       npt.NDArray[np.float32]

    _attribute_inputs:  npt.NDArray[np.float32]

    _inner_outputs:     npt.NDArray[np.float32]
    _inner_errors:      npt.NDArray[np.float32]
    
    _output:            np.float32
    _error:             np.float32

    nodes_per_layer:    int
    layer_count:        int
    attribute_count:    int
    training_dat_count: int

    def __init__(
            self,
            attribute_count: int,
            layer_count: int, 
            perceptron_per_layer_count: int,
            training_dat_count: int
            ):

        self._output = np.float32(0)
        self._error = np.float32(0)

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
                shape=(self.nodes_per_layer,
                       attribute_count + 1),
                dtype=np.float32
                )

        # inner_weights[i, j, k] := weight from inner layer i node k
        # to inner layer i+1 node j
        # i.e. the row [i, j] denotes all the weights coming in to
        # node j.
        self.inner_weights = np.zeros(
                shape=(self.layer_count - 1,
                       self.nodes_per_layer,
                       self.nodes_per_layer + 1),
                dtype=np.float32
                )

        # exit_weights[i] := weight from inner layer n-1 node i
        # to result layer node
        self.exit_weights = np.zeros(
                shape=(self.nodes_per_layer + 1,),
                dtype=np.float32
                )

        self._attribute_inputs = np.empty(
                shape=(self.attribute_count + 1,),
                dtype=np.float32
                )
        
        self._inner_outputs = np.empty(
                shape=(self.layer_count, self.nodes_per_layer),
                dtype=np.float32
                )



    def forward(self, x: npt.NDArray[np.float32])->np.float32:
        '''returns a val in (0, 1) from input with dummy 1 attribute at the end
        
        Parameters
        ----------
        x : npt.NDArray[np.float32]
            A list of attributes

        Returns
        -------
        np.float32
            Sigmoid prediction value
        '''
        
        self._attribute_inputs = np.append(1, x)
        
        # Size logic (obviously there's the sigmoid, but let's not care)
        #
        # Let n := attribute_count, m:= node/layer, l := layer count
        #
        # Let   self.attribute_list = (n,)
        #       self.inner_weights  = (l, m, m - 1)
        #       self.enter_weights  = (n, m - 1)
        #
        # So    self._inner_outputs[0]
        #           = (self.attribute_list @ self.enter_weights)
        #           = ((n,) @ (n, m)) = (1, m)
        # And   self._inner_outputs[i+1]
        #           = (self.inner_weights[i].append(1) @ self._inner_outputs[i])
        #           = ((m, m+1) @ (m+1,)).append(1)
        #           = (m,)
        # And   self._output
        #           = (self._inner_outputs[l-1].append(1) @ self.exit_weights)
        #           = ((m,) @ (m,))
        #           = constant (or I suppose so?)

        self._inner_outputs[0] = self.sigmoid(
                self.enter_weights @ self._attribute_inputs)
        print(self._inner_outputs[0])
        
        for i in range(0, self.layer_count - 1):
            layer_input = np.append(1, self._inner_outputs[i])
            layer_weights = self.inner_weights[i]
            try:
                self._inner_outputs[i + 1] = self.sigmoid(
                        layer_weights @ layer_input)
            except:
                exit(0)

        

        output_matrix = self.exit_weights @ np.append(
                1, self._inner_outputs[self.layer_count - 1])
        self._output = self.sigmoid(output_matrix).item()
        print(self._output)
        return self._output

    def sigmoid(self, x: np.ndarray)->np.ndarray:
        return 1 / (1 + np.exp(-x))


def main():
    '''For debug only
    '''
    nn = NeuralNetwork(attribute_count= 2, layer_count= 3, perceptron_per_layer_count= 2, training_dat_count= 1)
    nn.enter_weights = np.array([
        [1, 2, 1.5],
        [-1, 0, -0.5]
        ], dtype=np.float32)
    print(nn.enter_weights)
    nn.inner_weights = np.array([
            [
                [3, 3, 3.3],
                [4, 4, 4.4]
            ],
            [
                [10, 10, 10.1],
                [-2, -3, -15.5]
            ]
        ], dtype=np.float32)
    print(nn.inner_weights)
    nn.exit_weights = np.array([
        5, 5.5, 5.55
        ], dtype=np.float32)
    print(nn.exit_weights)
    print("--------------------------------------------")
    print(nn.forward(np.array([1, 2], dtype=np.float32)))


if __name__ == '__main__':
    main()
