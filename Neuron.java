import java.util.ArrayList;
import java.util.List;

public class Neuron {
    double value;
    double neuronErr;

    public List<Double> weights = new ArrayList<>();
    public List<Double> gradWeights = new ArrayList<>();
    public List<Double> deltaWeights = new ArrayList<>();

    public Neuron() {

    }

    // GETTERS!!!

    public double getValue() {
        return value;
    }

    public double getWeight(int index) {
        return weights.get(index);
    }

    public double getGradWeight(int index) {
        return gradWeights.get(index);
    }

    public double getNeuronErr() {
        return neuronErr;
    }

    public double getDeltaWeight(int index) {
        return deltaWeights.get(index);
    }


    //ADD FUNCTIONS!!!

    public void addWeight(double value) {
        weights.add(value);
    }

    public void addDeltaWeight(double value) {
        deltaWeights.add(value);
    }

    public void addGradWeight(double value) {
        gradWeights.add(value);
    }
    

    //SETTERS!!!

    public void setWeight(int index, double value) {
        weights.set(index, value);
    }

    public void setGradWeight(int index, double value) {
        gradWeights.set(index, value);
    }

    public void setValue(double value) {
        this.value = value;
    }

    public void setNeuronErr(double neuronErr) {
        this.neuronErr = neuronErr;
    }

    public void setDeltaWeight(int index, double value) {
        deltaWeights.set(index, value);
    }
}
