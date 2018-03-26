import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Net {
    private List<List<Neuron>> net = new ArrayList<>();
    private double target;
    private final double eta = 0.1;
    private final double momentum = 0.5;
    Random rand = new Random();

    public Net() {
        init();
        for (int i = 0; i < 10000; i++) {
            updatingInputs();
            System.out.println();

            feedForward();
            System.out.println();

            calcErr();
            System.out.println();

            calsWeightsGrad();
            System.out.println();

            updateWeights();
            System.out.println();
        }
    }
    public void init() {
        createTopology(2);
        createTopology(3);
        createTopology(2);
        createTopology(1);

        int neuronsInNextLayer = 0;
        for (int i = 0; i < net.size() - 1; i++) {
            neuronsInNextLayer = net.get(i + 1).size();
            for (int j = 0; j < net.get(i).size(); j++) {
                for (int k = 0; k < neuronsInNextLayer - 1; k++) { // BIAS NEURON!!!
                    net.get(i).get(j).addWeight(Math.random());
                    net.get(i).get(j).addDeltaWeight(0);
                    net.get(i).get(j).addGradWeight(0);
                    //System.out.println("Weight for neuron [" + i + "][" + j + "][" + k + "] -> "
                      //      + net.get(i).get(j).getWeight(k));
                }
            }
        }

        // For BIAS NEURONS!!!;
        for (int i = 0; i < net.size(); i++) {
            net.get(i).get(net.get(i).size() - 1).setValue(1.0);
        }
    }

    public void updatingInputs() {
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < net.get(i).size() - 1; j++) { // BIAS NEURON!!!
                net.get(i).get(j).setValue(rand.nextInt(2));
                System.out.println("The input value of neuron [" + i + "][" + j + "] -> " + net.get(i).get(j).getValue());
            }
        }

        double n1 = net.get(0).get(0).getValue(); // TO create XOR operation
        double n2 = net.get(0).get(1).getValue(); // TO create XOR operation

        if (n1 == 1.0 && n2 == 0.0 || n1 == 0.0 && n2 == 1.0) {
            target = 1.0;
        } else {
            target = 0.0;
        }

        System.out.println("Target of net -> " + target);
    }

    public void feedForward() {
        double sum = 0;
        int codeKey;
        for (int i = 0; i < net.size(); i++) {
            codeKey = 0;
            int nL = i == net.size() - 1 ? 0 : net.get(i + 1).size() - 1; // nL == "Next Layer" // FOR BIAS NEURON!!!
            for (int j = 0; j < nL; j++) {
                for (int k = 0; k < net.get(i).size(); k++) { // For BIAS NEURON!!!
                    sum += net.get(i).get(k).getWeight(codeKey) * net.get(i).get(k).getValue();
                }
                net.get(i + 1).get(j).setValue(actF(sum));
                codeKey++;
                sum = 0;

                System.out.println("Neuron [" + (i + 1) + "][" + j + "] -> " + net.get(i + 1).get(j).getValue());
            }
        }
    }

    public void calcErr() {
        double sum = 0;
        int codeKey;
        for (int i = net.size() - 1; i > 1; i--) {
            if (i == net.size() - 1) {
                net.get(i).get(0).setNeuronErr((target - net.get(i).get(0).getValue()) * actFDeriv(net.get(i).get(0).getValue()));

                System.out.println("The main error [" + i + "][" + 0 + "] -> " + (target - net.get(i).get(0).getValue())); // The error to OUTPUT
            }
            int pL = net.get(i - 1).size() - 1; // pL == "Previous Layer" // FOR BIAS NEURON!!!
            for (int j = 0; j < pL; j++) {
                codeKey = 0;
                for (int k = 0; k < net.get(i).size() - 1; k++) { // FOR BIAS NEURON!!!
                    sum += net.get(i - 1).get(j).getWeight(codeKey++) * net.get(i).get(k).getNeuronErr();
                }
                net.get(i - 1).get(j).setNeuronErr(sum * actFDeriv(net.get(i - 1).get(j).getValue()));
                sum = 0;

                //System.out.println("The error of neuron [" + (i - 1) + "][" + j + "] -> " + net.get(i - 1).get(j).getNeuronErr());
            }
        }
    }

    public void calsWeightsGrad() {
        double sum;
        int codeKey = 0;
        for (int i = 0; i < net.size() - 1; i++) {
            int nL = i == net.size() - 1 ? 0 : net.get(i + 1).size(); // nL == "Next Layer"
            for (int j = 0; j < net.get(i).size(); j++) {
               for (int k = 0; k < net.get(i).get(j).gradWeights.size(); k++) {
                   sum = net.get(i).get(j).getValue() * net.get(i + 1).get(codeKey).getNeuronErr();
                   net.get(i).get(j).setGradWeight(k, sum);
                   codeKey++;

                   //System.out.println("Gradient weight for neuron [" + i + "][" + j + "][" + k + "] -> " + net.get(i).get(j).getGradWeight(k));
               }
               codeKey = 0;
            }
        }
    }

    public void updateWeights() {
        for (int i = 0; i < net.size() - 1; i++) {
            int nL = i == net.size() - 1 ? 0 : net.get(i + 1).size();
            for (int j = 0; j < net.get(i).size(); j++) {
                for (int k = 0; k < net.get(i).get(j).weights.size(); k++) {
                    double oldDelta = net.get(i).get(j).getDeltaWeight(k);
                    double newDelta = oldDelta * momentum + eta * net.get(i).get(j).getGradWeight(k);
                    net.get(i).get(j).setDeltaWeight(k, newDelta);
                    net.get(i).get(j).setWeight(k, (net.get(i).get(j).getWeight(k) + newDelta));

                    // NEED TO ADD SOME SOUT TO THIS
                }
            }
        }
    }

    public double actF(double x) {
        return Math.tanh(x);
    }

    public double actFDeriv(double f) {
        return 1.0 - f * f;
    }

    public void createTopology(int countOfNeurons) {
        List<Neuron> temp = new ArrayList<>();
        for (int i = 0; i < countOfNeurons; i++) {
            temp.add(new Neuron());
        }
        temp.add(new Neuron()); // BIAS NEURON!!!
        net.add(temp);
    }
}