package test;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import ldaCore.OnlineLDA2;



public class ExecuteFormattedInputfile{
	// Container
	static ArrayList<String> fileNames = new ArrayList<String>();
	static ArrayList<String> topicGroup= new ArrayList<String>();
	static ArrayList<Integer> gammaTopics= new ArrayList<Integer>();
	static ArrayList<String[][]> stringBatchList = new ArrayList<String[][]>();
	static HashMap<String, ArrayList<Integer>> topicGammaTopicMap = new HashMap<String, ArrayList<Integer>>();

	// Constant Parameters
	static int batchSize_ = 10;
	static String targetURI;
	
	// limit Read File Per Directory
//	static int limit = 10;

	// LDA Parameters
	static int K = 4;
	static double alpha = 1./(K);
	static double eta = 1./ (K);
	static double tau0 = 80;	// 1024
	static double kappa = 0.8;	// 0.7
	static int IterNum = 100;
	static int PPLNUM = 3;
	static int totalD= (int)11000;

	// Control
	static int limit = 10000;
	static int trainLine=10000;
	
	static OnlineLDA2 onlineLDA2;

	static String stopWord = "a b c d e f g h i j k l m n o p q r s t u v w x y z the of in and have to it was or were this that with is some on for so how you if would com be your my one not never then take for an can no but aaa when as out just from does they back up she those who another her do by must what there at very are am much way all any other me he something someone doesn his also its has into us him than about their may too will had been we them why did being over without these could out which only should even well more where after while anyone our now such under two ten else always going either each however non let done ever between anything before every same since because quite sure here nothing new don off still down yes around few many own go get know think like make say see look use said";
	static String[] stopWords;
	
	public static void main(String[] args){
		long start = System.nanoTime();
		targetURI = "/Users/ishikawanaoki/dataset/reuters_count.txt";
		
	
		stopWords = stopWord.split(" ");	
		onlineLDA2 = new OnlineLDA2(K, alpha, eta, totalD, tau0, kappa, batchSize_, stopWord);
		
		try {
			executeTraining();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		long end = System.nanoTime();
		
//		printConfusionMatrix();
		onlineLDA2.showTopicWords();
		
		System.out.println("Experiment Time:" + (end - start));
	}

	private static void executeTraining() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(targetURI));
		ArrayList<String[]> miniBatchArrayList = new ArrayList<String[]>();
		String[][] miniBatch;	// TODO
		
		int time = 0;
		while(true){
			String docString = br.readLine();
			time++;
			
			if(docString == null){
				executeMiniBatchLearining(miniBatchArrayList, time);
				break;
			}else{
				String[] labelValues = docString.split(" "); 
				miniBatchArrayList.add(labelValues);

				if(miniBatchArrayList.size() == batchSize_){
					System.out.println("time:" +time);
					executeMiniBatchLearining(miniBatchArrayList, time);
					miniBatchArrayList.clear();
				}
			}
		}
	}
	



	private static void executeMiniBatchLearining(ArrayList<String[]> miniBatchArrayList, int time) {
		String[][] miniBatch = new String[miniBatchArrayList.size()][];
		for(int d=0, SIZE = miniBatchArrayList.size(); d< SIZE; d++){
			miniBatch[d] = miniBatchArrayList.get(d);
		}
		onlineLDA2.trainMiniBatch(miniBatch, time);
	}

	private static String[] processLine(String line) {
		
		String stopWord = "a b c d e f g h i j k l m n o p q r s t u v w x y z the of in and have to it was or were this that with is some on for so how you if would com be your my one not never then take for an can no but aaa when as out just from does they back up she those who another her do by must what there at very are am much way all any other me he something someone doesn his also its has into us him than about their may too will had been we them why did being over without these could out which only should even well more where after while anyone our now such under two ten else always going either each however non let done ever between anything before every same since because quite sure here nothing new don off still down yes around few many own go get know think like make say see look use said";
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
			ret[0] = "XXXXXXXXXXXXXXX";
			return ret;
		}
		
		ArrayList<String> tmpArrayList = new ArrayList<String>();
		for(int i=0; i<ret.length; i++){
			if(ret[i].length() >= 0 && !stopWordSet.contains(ret[i])){
				tmpArrayList.add(ret[i]);
			}
		}
		
		String[] ret2 = new String[tmpArrayList.size()];
		for(int i=0, Size=tmpArrayList.size(); i<Size; i++){
			ret2[i] = tmpArrayList.get(i);
		}

		return ret2;
	}
}
