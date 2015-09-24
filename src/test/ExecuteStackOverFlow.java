package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;

import org.apache.commons.math3.special.Gamma;

import ldaCore.OnlineLDA2;

public class ExecuteStackOverFlow {
	
	// Container
	static ArrayList<String> fileNames = new ArrayList<String>();
	static ArrayList<String[][]> stringBatchList = new ArrayList<String[][]>();

	// Free Variables
	private static String baseURI = " /Users/ishikawanaoki/Documents/datasetML/stackoverflow/";
//	private static String baseURI = "/Users/ishikawanaoki/Documents/workspace/LDA/myCorpus/";
	private static int batchSize_ = 10;
	private static int K = 6;
	private static double alpha = 1. / K;
	private static double eta   = 1. / (K);
	private static int totalD   = 11000;
	private static double tau0  = 1000;
	private static double kappa = 0.9;
	
	// Control
	private static int PPLNUM = 2000;
	private static int trainLine = (int)1E4;
	
	static OnlineLDA2 onlineLDA2;
	public static void main(String[] args) {
		
		// IMPORT FILE
		getFiles();


		String stopWord = "a b c d e f g h i j k l m n o p q r s t u v w x y z the of in and have to it was or were this that with is some on for so how you if would com be your my one not never then take for an can no but aaa when as out just from does they back up she those who another her do by must what there at very are am much way all any other me he something someone doesn his also its has into us him than about their may too will had been we them why did being over without these could out which only should even well more where after while anyone our now such under two ten else always going either each however non let done ever between anything before every same since because quite sure here nothing new don off still down yes around few many own go get know think like make say see look use said";
		// generate LDA
		onlineLDA2 = new OnlineLDA2(K, alpha, eta, totalD, tau0, kappa, batchSize_, stopWord);

		// MAKE BATCH
		try {
			executeTraining();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		onlineLDA2.showTopicWords();
		
//		// Train *************************************************
//		int time = 0;
//		System.out.println("Time MAX:" + stringBatchList.size() * batchSize_);
//		
//		// FOR PERPLEXITY LOOP
//		for(int ppl=0; ppl<PPLNUM; ppl++){
//			int tmp = 0;
//			for(String[][] stringBATCH:stringBatchList){
//				//
////				for(int d=0; d<stringBATCH.length; d++){
////					for(int w=0; w<stringBATCH[d].length; w++){
////						if(stringBATCH[d][w].length()==0){
////							System.out.println("====X X X X X X X X X X X X====");
////						}
////					}
////				}
//				//
//				tmp++;
//				if(tmp >=30)
//					break;
//				onlineLDA2.trainMiniBatch(stringBATCH, time);
//				time += batchSize_;
//			}
//			onlineLDA2.showTopicWords();
////			System.out.print(/*"perplexity:" +*/ onlineLDA2.getPerplexity() + ",");
////			System.out.print(onlineLDA_Batch.getBound() + ",");
//		}	
	}

	private static void executeTraining() throws IOException {

		String URI = "/Users/ishikawanaoki/Documents/datasetML/stackoverflow/stackoverflow.lda.vw";
		BufferedReader br = new BufferedReader(new FileReader(URI));

		ArrayList<String[]> tmpBatchList= new ArrayList<String[]>();

		int tmp = 0;

		String[][] tmpMiniBatch;

		while(true){
			// Line Limitation
			if(tmp > trainLine){
				break;
			}

			if(tmp%1000 ==0){
//				System.out.println((tmp*1. / trainLine) * 100 + "%");
			}
			
			// Get Line 
			String line = br.readLine();
			if(line==null)
				break;
			String[] words = processLine(line);
			tmpBatchList.add(words);
			
			// train
			if(tmpBatchList.size() == batchSize_){
				tmpMiniBatch = new String[batchSize_][];
				for(int d=0; d<batchSize_; d++){
					tmpMiniBatch[d] = tmpBatchList.get(d);
				}
				onlineLDA2.trainMiniBatch(tmpMiniBatch, tmp);
				tmpBatchList.clear();
			}
			
			if(tmp%100 == 0){
//				System.out.printf("%.2f,",onlineLDA2.getPerplexity());
				System.out.println(tmp);
			}

			// INCREMENT
			tmp++;
		}
		br.close();
		
		// SHOW
		onlineLDA2.showTopicWords();
		
	}
	

	private static String[] processLine(String line) {
		
		String stopWord = "a b c d e f g h i j k l m n o p q r s t u v w x y z the of in and have to it was or were this that with is some on for so how you if would com be your my one not never then take for an can no but aaa when as out just from does they back up she those who another her do by must what there at very are am much way all any other me he something someone doesn his also its has into us him than about their may too will had been we them why did being over without these could out which only should even well more where after while anyone our now such under two ten else always going either each however non let done ever between anything before every same since because quite sure here nothing new don off still down yes around few many own go get know think like make say see look use said using file";
		String[] stopWords = stopWord.split(" ");
		HashSet<String> stopWordSet = new HashSet<String>();
		for(String tmpStopWord: stopWords){
			stopWordSet.add(tmpStopWord);
		}

		line = line.toLowerCase();
		
		line = line.replace("\"", " ");
		line = line.replace("\\", " ");
		line = line.replace("/", " ");
		line = line.replace(">", " ");
		line = line.replace("<", " ");
		line = line.replace("-", " ");
		line = line.replace(",", " ");
		line = line.replace(".", " ");
		line = line.replace("(", " ");
		line = line.replace(")", " ");
		line = line.replace(":", " ");
		line = line.replace(";", " ");
		line = line.replace("'", " ");
		line = line.replace("[", " ");
		line = line.replace("]", " ");
		line = line.replace("!", " ");
		line = line.replace("*", " ");
		line = line.replace("#", " ");
		line = line.replace("+", " ");
		line = line.replace("%", " ");
		line = line.replace("@", " ");
		line = line.replace("&", " ");
		line = line.replace("?", " ");
		line = line.replace("$", " ");
		line = line.replace("0", " ");
		line = line.replace("1", " ");
		line = line.replace("2", " ");
		line = line.replace("3", " ");
		line = line.replace("4", " ");
		line = line.replace("5", " ");
		line = line.replace("6", " ");
		line = line.replace("7", " ");
		line = line.replace("8", " ");
		line = line.replace("9", " ");
		line = line.replace("\t", " ");
		line = line.replace("_", " ");
		line = line.replace("{", " ");
		line = line.replace("}", " ");
		line = line.replace("=", " ");
		line = line.replace("|", " ");
		
		
		String[] ret;
		
		
		ret = line.split(" ");
		
		if(line.equals("")){
			ret = new String[1];
			ret[0] = "a";
			return ret;
		}
		
		
		for(int i=0; i<ret.length; i++){
//			ret[i] = ret[i].replace(" ", "");
//			if(ret[i].length() <= 4){
//				ret[i] = "b";
//			}
		}
		
		ArrayList<String> tmpArrayList = new ArrayList<String>();
		for(int i=0; i<ret.length; i++){
			if(ret[i].length() != 0 && !stopWordSet.contains(ret[i])){
				tmpArrayList.add(ret[i]);
			}
		}
		
		String[] ret2 = new String[tmpArrayList.size()];
		for(int i=0, Size=tmpArrayList.size(); i<Size; i++){
			ret2[i] = tmpArrayList.get(i);
		}
		
		// TODO remove
//		System.out.println("ret2.length:" + ret2.length);

		return ret2;
	}	

	private static void getFiles() {
		
		File tmpFile = new File(baseURI);
		String[] names= tmpFile.list();

//		for(int i=0; i<names.length; i++){
//			fileNames.add(baseURI + names[i]);	
//		}
		fileNames.add("/Users/ishikawanaoki/Documents/datasetML/stackoverflow/stackoverflow.lda.vw");
	}
}
