package ldaCore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

public class OnlineLDA {
	
	private int K_;	// number of Cluster
	private double rhot_;	// rhot
	private double tau0_;	// 
	private double kappa_;	// 
	private double eta0_;	// 
	private double alpha_;
	private ArrayList<HashMap<String, double[]>> phi_;
	private double[][] gamma_;
	private double[][] ElogTheta;
	private double[][] expElogTheta;
	private Map<String, double[]> lambda_;
	private Map<String, Double>[] lambdaBar_;
	private Map<String, Double>[] vacabulary;
	
	// Random
	private final double gammaShape = 10;
	private final double gammaScale = 0.1;
	GammaDistribution gd = new GammaDistribution(gammaShape, gammaScale);
	
	private int D_;
	private HashMap<String, Double> tmpVocabulary;	
	private ArrayList<HashMap<String, Double>> tmpVocabularyPerDoc;	
	private HashMap<String, Double> vocabulary;	// TODO initialize
	
	private String[][] wordids;
	private double[][] wordcts;
	
	GammaDistribution gamma = new GammaDistribution(gammaShape, gammaScale);
	
	// constructor
	public OnlineLDA(int k, double tau0, double kappa, double eta0, double alpha){
		K_ = k;
		tau0_ = tau0;
		kappa_ = kappa;
		eta0_  = eta0;
		alpha_ = alpha;
	}
	
	public void trainPerBatch(String[][] stringBatch, int time){
		// check New Word
		D_ = stringBatch.length;
		checkNewWords(stringBatch);
		
		// 
		do_e_step(stringBatch);	
		
		//
		

			
	}

	private void checkNewWords(String[][] stringBatch) {
		// TODO Auto-generated method stub
		tmpVocabulary= new HashMap<String, Double>();
		tmpVocabularyPerDoc = new ArrayList<HashMap<String,Double>>();

		for(int d=0; d<D_; d++){
			HashMap<String, Double> tmpVocabulary_Doc = new HashMap<String, Double>();
			for(int w=0, Nd=stringBatch[d].length; w<Nd; w++){
				String tmpString = stringBatch[d][w];
				
				// tmpVocabulary_Doc
				if(!tmpVocabulary_Doc.containsKey(tmpString)){
					tmpVocabulary_Doc.put(tmpString, 1d);
				}else{
					double tmpCnt = tmpVocabulary_Doc.get(tmpString);
					tmpCnt++;
					tmpVocabulary_Doc.put(tmpString, tmpCnt);
				}

				// tmpVocabulary for Batch
				if(!tmpVocabulary.containsKey(tmpString)){
					tmpVocabulary.put(tmpString, 1d);
				}else{
					double tmpCnt = tmpVocabulary.get(tmpString);
					tmpCnt++;
					tmpVocabulary.put(tmpString, tmpCnt);
				}
				
				// Vocabulary
				if(!vocabulary.containsKey(tmpString)){
					vocabulary.put(tmpString, 1d);
				}else{
					double tmpCnt = vocabulary.get(tmpString);
					tmpCnt++;
					vocabulary.put(tmpString, tmpCnt);
				}
			}
			tmpVocabularyPerDoc.add(tmpVocabulary_Doc);
		}
	}

	private void do_e_step(String[][] stringBatch) {
		// prepare
		wordids = new String[D_][];
		wordcts = new double[D_][];
		getWordidsAndCts();

		gamma_ = getGamma();
		ElogTheta = dirichlet_distribution(gamma_);
		expElogTheta = calcExpMatrix(ElogTheta);
		
		double[][] sstats = new double[lambda_.size()][K_];
		for(int i=0; i<lambda_.size(); i++){
			Arrays.fill(sstats[i], 0d);
		}
		
		for(int d=0; d<D_; d++){
			String[] ids = wordids[d];
			double[] cts = wordcts[d];

			double[] gammad = gamma_[d];
			double[] Elogthetad = ElogTheta[d];
			double[] expElogthetad = expElogTheta[d];
			double[][] expElogbetad = getElogbetad(ids);
			
			double[] phinorm = dot(expElogthetad, expElogbetad);
			
			
		}
	}
	
	private double[] dot(double[] expElogthetad, double[][] expElogbetad) {
		int wordSize = expElogbetad[0].length;
		double[] ret = new double[wordSize];
		for(int w=0, Nd=wordSize; w<Nd; w++){
			double tmp = 0;
			for(int k=0; k<K_; k++){
				tmp += expElogthetad[k] * expElogbetad[k][w];
			}
			ret[w] = tmp;
		}
		return ret;
	}

	private double[][] getElogbetad(String[] ids) {
		// returns K * Nd matrix
		int SIZE = ids.length;
		double[][] ret = new double[K_][SIZE];
		int tmpIdx = 0;
		for(String tmpStr:ids){
			for(int k=0; k<K_; k++){
				ret[k][tmpIdx] = ElogBeta.get(tmpStr)[k];
			}
			tmpIdx++;
		}
		return ret;
	}

	private void getWordidsAndCts() {
		for(int d=0; d<D_; d++){
			int tmpWordSize = tmpVocabularyPerDoc.size();
			String[] tmpString_Doc = new String[tmpWordSize];
			double[] tmpCounts_Doc  = new double[tmpWordSize];

			int tmpIdx = 0;
			HashMap<String, Double> tmpVocabulary_Doc = tmpVocabularyPerDoc.get(d);
			for(String tmpStr:tmpVocabulary_Doc.keySet()){
				tmpString_Doc[tmpIdx] = tmpStr;
				tmpCounts_Doc[tmpIdx]  = tmpVocabulary_Doc.get(tmpStr); 
				tmpIdx++;
			}
			wordids[d] = tmpString_Doc;
			wordcts[d] = tmpCounts_Doc;
		}
	}

	private double[][] calcExpMatrix(double[][] elogTheta2) {
		double[][] ret = new double[elogTheta2.length][elogTheta2[0].length];
		for(int i=0; i<elogTheta2.length; i++){
			for(int j=0; j<elogTheta2[0].length; j++){
				ret[i][j] = Math.exp(elogTheta2[i][j]);
			}
		}
		return ret;
	}

	private double[][] dirichlet_distribution(double[][] tmpMatrix) {
		int tmpD = tmpMatrix.length;
		int tmpK = tmpMatrix[0].length;
		double[][] ret = new double[tmpD][];
		
		for(int d=0; d<tmpD; d++){
			double tmpSum = 0;
			for(int k=0; k<tmpK; k++){
				tmpSum += tmpMatrix[d][k];
			}
			tmpSum = Gamma.digamma(tmpSum);
			
			for(int k=0; k<K_; k++){
				ret[d][k] = Gamma.digamma(tmpMatrix[d][k]) - tmpSum;
			}
		}
		return ret;
	}

	private double[][] getGamma() {
		double[][] ret = new double[D_][K_];
		for(int d=0; d<D_; d++){
			for(int k=0; k<K_; k++){
				ret[d][k] = gd.sample();
			}
		}
		return ret;
	}
}













