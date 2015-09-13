package ldaCore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;


public class OnlineLDA2 {
	// 
	private Map<String, Integer> vocab = new HashMap<String, Integer>();
	
	// Free Parameters
	private int K_;	
	
	// 
	private int D_;	
	private int totalD_ = 1024;
	private int[] Nds;
	
	//
	double[][][] phi_;
	double[][]   gamma_;
	HashMap<Integer, double[]> lambda_; 
	
	//
	private int[][] ids;	
	private int[][] cts;
	
	// Random Variables
	GammaDistribution gd;
	
	// CONSTANTS
	private double SHAPE = 100;
	private double SCALE = 1 / SHAPE;
	
	private double DELTA = 1E-6;
	
	private double tau0_ = 1024;
	private double kappa_= 0.7;
	private double rhot;
	private double alpha_;
	private double eta_;
	
	//
	private ArrayList<String> newWords;
	
	// Constructor
	public OnlineLDA2(int K, double alpha, double eta, int totalD){
		// Initialize Free Params
		K_ = K;
		alpha_ = alpha;
		eta_ = eta;
		totalD_ = totalD;
		
		// Initialize Internal Params
		setRandomParams();
		setParams();
	}
	
	
	private void setParams() {
		lambda_ = new HashMap<Integer, double[]>();
	}


	private void setRandomParams() {
		gd = new GammaDistribution(SHAPE, SCALE);
	}


	public void getIdsCts(String[][] miniBatch){
		// global
		ids = new int[D_][];
		cts = new int[D_][];
		
		newWords = new ArrayList<String>();
		
		
		// HashMap of word per Documents
		Map<Integer, Integer> wordCountMap = new HashMap<Integer, Integer>();
		
		for(int d=0; d<D_; d++){
			// get Nd : number of word per documents
			int Nd = miniBatch[d].length;
			// reset wordCountMap
			wordCountMap = new HashMap<Integer, Integer>();

			for(int w=0; w<Nd; w++){
				String tmpWord = miniBatch[d][w];
				if(!vocab.containsKey(tmpWord)){
					vocab.put(tmpWord, vocab.size());
					newWords.add(tmpWord);
				}
				
				int tmpId = vocab.get(tmpWord);
				if(!wordCountMap.containsKey(tmpId)){
					wordCountMap.put(tmpId, 1);
				}else{
					int tmpFrequency = wordCountMap.get(tmpId);
					tmpFrequency++;
					wordCountMap.put(tmpId, tmpFrequency);
				}
			}
			updateIdCts(d, wordCountMap);
		}
	}

	private void updateIdCts(int d, Map<Integer, Integer> wordCountMap) {
		int[] tmpIds = new int[wordCountMap.size()];
		int[] tmpCts = new int[wordCountMap.size()];
		
		int tmpId = 0;
		for(int id:wordCountMap.keySet()){
			tmpIds[tmpId] = id;
			tmpIds[tmpId] = wordCountMap.get(id);
			
			tmpId++;
		}
		
		ids[d] = tmpIds;
		cts[d] = tmpCts;
	}

	public void trainMiniBatch(String[][] miniBatch, int time){
		// get the number of words(Nd) for each documents
		getMiniBatchParams(miniBatch);
		// get ids and cts
		getIdsCts(miniBatch);
		
		// check
		updateSizeOfParameterPerMiniBatch(miniBatch);

		rhot = Math.pow(tau0_+time, -kappa_);

		for(int d=0; d<D_; d++){
			// get ids and cts
			do_e_step(d);

			// update Lambda
			do_m_step(d);
		}
	}

	private void getMiniBatchParams(String[][] miniBatch) {
		//
		D_ = miniBatch.length;
		Nds = new int[D_];
		
		for(int d=0; d<D_; d++){
			Nds[d] = miniBatch[d].length;
		}
	}

	private void updateSizeOfParameterPerMiniBatch(String[][] miniBatch) {
		// phi_
		phi_ = new double[D_][][];
		gamma_ = new double[D_][];
		// gamma, lambda: None

		// gamma and phi
		for(int d=0; d<D_; d++){
			// Gamma
			double[] gammad = getRandomGammaArray();
			// phi_ not needed to be initialized
			double[][] phid = new double[Nds[d]][K_];
			
			gamma_[d] = gammad;
			phi_[d]   = phid;
		}
		
		// lambda
		for(String newWord: newWords){
			int tmpId = vocab.get(newWord);
			double[] lambdaNW = getRandomGammaArray(); 
			lambda_.put(tmpId, lambdaNW);
		}
	}

	private double[] getRandomGammaArray() {
		double[] ret = new double[K_];
		for(int k=0; k<K_; k++){
			ret[k] = gd.sample();
		}
		return ret;
	}

	private void do_m_step(int d) {
		int vSIZE = lambda_.size();
		
		HashMap<Integer, double[]> lambdaBar = new HashMap<Integer, double[]>();
		
		double lambdaBar_kw = 0;

		for(int k=0; k<K_; k++){
			for(int w=0; w<Nds[d]; w++){
				int tmpId = ids[d][w];
				lambdaBar_kw = eta_ + (totalD_ / D_) * cts[d][tmpId] + phi_[d][w][k];
				if(!lambdaBar.containsKey(tmpId)){
					double[] tmpDArray = new double[K_];
					Arrays.fill(tmpDArray, 0d);
					tmpDArray[k] = lambdaBar_kw;
					lambdaBar.put(tmpId, tmpDArray);
				}else{
					double[] tmpDArray= lambdaBar.get(tmpId);
					tmpDArray[k] += lambdaBar_kw;
					lambdaBar.put(tmpId, tmpDArray);
				}
			}
		}
		
		double oneMinusRhot = 1 - rhot;
		for(int lambdaKey:lambda_.keySet()){
			if(!lambdaBar.containsKey(lambdaKey)){
				double[] lambdaW = lambda_.get(lambdaKey);
				for(int k=0; k<K_; k++){
					lambdaW[k] = oneMinusRhot * lambdaW[k];
				}
				lambda_.put(lambdaKey, lambdaW);
			}else{
				double[] lambdaW = lambda_.get(lambdaKey);
				double[] lambdaBarW = lambdaBar.get(lambdaKey);
				for(int k=0; k<K_; k++){
					lambdaW[k] = oneMinusRhot * lambdaW[k] + rhot * lambdaBarW[k];
				}
				lambda_.put(lambdaKey, lambdaW);
			}
		}
	}

	private void do_e_step(int d) {
		double[] lastGamma;
		double[] nextGamma = copyArray(gamma_[d]);
		do{
			lastGamma = copyArray(nextGamma);
			double sumGammad = getSumDArray(gamma_[d]);
			double sumLambdaK;
			for(int k=0; k<K_; k++){
				sumGammad = getSumDArray(gamma_[d]);
				sumLambdaK= getSumLambdaK(d, k);

				// update phi
				for(int w=0; w<Nds[d]; w++){
					int tmpId = ids[d][k];
					double tmpEqtheta = Gamma.digamma(gamma_[d][k]) - Gamma.digamma(sumGammad);
					double tmpEqBeta  = Gamma.digamma(lambda_.get(tmpId)[k]) - Gamma.digamma(sumLambdaK);
					phi_[d][w][k] = Math.exp(tmpEqtheta + tmpEqBeta);
				}
				
				// update gamma
				double tmpGamma_dk = alpha_;
				for(int w=0; w<Nds[d]; w++){
					tmpGamma_dk += phi_[d][w][k] * cts[d][w];
				}
				gamma_[d][k] = tmpGamma_dk;
			}
		}while(!diffGamma(lastGamma, nextGamma));
	}


	private double getSumLambdaK(Integer d, int k) {
		double ret =0;
		for(int id:lambda_.keySet()){
			ret += lambda_.get(id)[k];
		}
		return ret;
	}


	private double getSumDArray(double[] ds) {
		double ret = 0;
		for(int k=0; k<K_; k++){
			ret += ds[k];
		}
		return ret;
	}


	private boolean diffGamma(double[] lastGamma, double[] nextGamma) {
		double diff = 0;
		for(int k=0; k<K_; k++){
			diff += Math.abs(lastGamma[k] - nextGamma[k]);
		}
		
		if(diff < K_ * DELTA){
			return true;
		}else{
			return false;
		}
	}

	private double[] copyArray(double[] ds) {
		double[] ret = new double[K_];
		for(int k=0; k<K_; k++){
			ret[k] = ds[k];
		}
		return ret;
	}
}
