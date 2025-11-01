import numpy        as np
import numpy.typing as npt
import math

class NeuralNetwork:    
    training_input:     npt.NDArray[np.float32]
    training_output:    npt.NDArray[np.float32]

    enter_weights:      npt.NDArray[np.float32] # (m-1, n)
    inner_weights:      npt.NDArray[np.float32] # (l, m-1, m)
    exit_weights:       npt.NDArray[np.float32] # (k, m)

    _attribute_inputs:  npt.NDArray[np.float32] # (n,)

    _inner_outputs:     npt.NDArray[np.float32] # (l, m-1)
    _inner_errors:      npt.NDArray[np.float32] # (l, m-1)
    
    _output:            npt.NDArray[np.float32] # (k)
    _error:             npt.NDArray[np.float32] # (k)

    nodes_per_layer:    int
    layer_count:        int
    attribute_count:    int
    training_dat_count: int
    class_count:        int

    lr:                 np.float32

    def __init__(
            self,
            attribute_count: int,
            layer_count: int, 
            perceptron_per_layer_count: int,
            training_dat_count: int,
            lr: float,
            class_count: int
            ):

        self.lr = np.float32(lr)

        self.nodes_per_layer    = perceptron_per_layer_count
        self.layer_count        = layer_count
        self.attribute_count    = attribute_count
        self.training_dat_count = training_dat_count
        self.class_count        = class_count

        self._output = np.empty(shape=(self.class_count,), dtype=np.float32)
        self._error  = np.empty(shape=(self.class_count,), dtype=np.float32)

        self.training_input = np.empty(
                shape=(training_dat_count, attribute_count),
                dtype=np.float32
                )
        self.training_output = np.empty(
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
                shape=(self.class_count, self.nodes_per_layer + 1,),
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
        self._inner_errors = np.empty(
                shape=(self.layer_count, self.nodes_per_layer,),
                dtype=np.float32
                )

    def forward(self, x: npt.NDArray[np.float32])->npt.NDArray[np.float32]:
        '''returns a val in (0, 1) from input without dummy 1 attribute at the end
        
        Parameters
        ----------
        x : npt.NDArray[np.float32]
            A list of attributes

        Returns
        -------
        np.float32
            Sigmoid prediction value
        '''
        
        self._attribute_inputs = np.append(arr=x, values=np.float32(1))
        
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
        #           = ((n,) @ (n, m-1)) = (1, m-1)
        # And   self._inner_outputs[i+1]
        #           = (self.inner_weights[i].append(1) @ self._inner_outputs[i])
        #           = ((m-1, m) @ (m,)).append(1)
        #           = (m-1,)
        # And   self._output
        #           = (self.exit_weights[n-1].append(1) @ self._inner_outputs[n-1])
        #           = ((k, m), (m,))
        #           = (k,)

        self._inner_outputs[0] = self.sigmoid(
                self.enter_weights @ self._attribute_inputs)
        
        for i in range(0, self.layer_count - 1):
            layer_input = np.append(arr=self._inner_outputs[i], values=1).astype(np.float32)
            layer_weights = self.inner_weights[i]
            self._inner_outputs[i + 1] = self.sigmoid(
                    layer_weights @ layer_input)

        last_input = np.append(arr=self._inner_outputs[self.layer_count-1], values=1).astype(np.float32)
        output_matrix = (self.exit_weights @ last_input).astype(np.float32)
        self._output = self.sigmoid(output_matrix)
        return self._output

    def backward(self, true_output: npt.NDArray[np.float32]):
        # Output error
        self._error = (self._output * (np.float32(1) - self._output) * (true_output - self._output)).astype(np.float32)
        
        # Latest hidden unit error

        o = np.append(arr=self._inner_outputs[self.layer_count - 1], values=1)  # (m,)
        w = self.exit_weights                                                   # (k, m)
        d = self._error                                                         # (k)
        last_inner_err = o * (1 - o) * (d @ w)                                  # (m,)
        self._inner_errors[self.layer_count - 1] = last_inner_err[:-1]          # (m-1,)

        # other inner errors

        for i in range(self.layer_count - 2, -1, -1):
            o = np.append(values=1, arr=self._inner_outputs[i]) # (m,)
            w = self.inner_weights[i]                           # (m-1, m)
            d = self._inner_errors[i+1]                         # (m-1,)
            err = o * (1 - o) * (d @ w)                         # (m,)
            self._inner_errors[i] = err[:-1]                    # (m-1,)
        
        # weight calculation as follow:
        # 
        # let in := size of layer l (with bias), out := size of layer l+1 (no bias)
        #
        # let w := weight[i]    (out, in)
        # let e := err[i+1]     (out,)
        # let o := output[i]    (in,)
        #
        # we calculate by:
        #   let E := (out, in), let each column be e
        #   let O := (out, in), let each row be o
        #   so w' := w + lr * E * O

        # calculate enter weights
        e = self._inner_errors[0]
        o = self._attribute_inputs
        grad = self.lr * np.outer(e, o)
        self.enter_weights = self.enter_weights + grad

        # calculate weights between hidden layers
        for i in range(0, self.layer_count - 1):
            e = self._inner_errors[i + 1]
            o = np.append(self._inner_outputs[i], 1)
            grad = self.lr * np.outer(e, o)
            self.inner_weights[i] = self.inner_weights[i] + grad
        
        # calculate exit weights
        e = self._error
        o = np.append(self._inner_outputs[self.layer_count - 1], 1)
        grad = self.lr * np.outer(e, o)
        self.exit_weights = (self.exit_weights + grad).astype(np.float32)

        pass

    def sigmoid(self, x: npt.NDArray[np.float32])->npt.NDArray[np.float32]:
        result = np.float32(1) / (np.float32(1) + np.exp(-x))
        return result.astype(np.float32)
    
    def randomize_weights(self):
        pass

##################################################################################


def main():
    '''For debug only
    '''
    pass


if __name__ == '__main__':
    main()
