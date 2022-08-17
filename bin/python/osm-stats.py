import networkx as nx
import cenpy
import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import time
from datetime import timedelta
import os
from joblib import Parallel, delayed
import multiprocessing
#test
ox.settings.timeout=100000

# https://stackoverflow.com/a/4676482/3023033
def flatten(TheList):
	listIsNested = True
	while listIsNested:                 #outer loop
		keepChecking = False
		Temp = []
		for element in TheList:         #inner loop
			if isinstance(element,list):
				Temp.extend(element)
				keepChecking = True
			else:
				Temp.append(element)
		listIsNested = keepChecking     #determine if outer loop exits
		TheList = Temp[:]
	return TheList

def get_msa_municipality_list_df(save=True):
	msa_df = pd.read_csv('../../data/raw/metrolist.csv', engine='python', header=None, skipfooter=5)
	for i in range(len(msa_df[[3]])):
		if '+' in msa_df.iloc[i,3]:
			names = msa_df.iloc[i,3].split(' ')
			state = ', ' + names[-1]
			if names[0] == 'Fairfax,':
				county = names[0][:-1] + ' County' + state
				city_1 = names[1] + state           
				city_2 = names[4] + ' ' + names[5][:-1] + state
				county_city_list = [county, city_1, city_2]            
			elif names[0] == 'Prince':
				if names[1] == 'William,':
					county = names[0] + ' ' + names[1][:-1] + ' County' + state
					city_1 = names[2] + state
					city_2 = names[4] + ' ' + names[5][:-1] + state
					county_city_list = [county, city_1, city_2]
				else:
					county = names[0] + ' ' + names[1] + ' County' + state
					city = names[3][:-1] + state
					county_city_list = [county, city]
			elif names[0] == 'James':
				county = names[0] + ' ' + names[1] + ' County' + state
				city = names[3] + state
				county_city_list = [county, city]
			elif names[0] == 'Dinwiddie,':
				county = names[0][:-1] + ' County' + state
				city_1 = names[1] + ' ' + names[2] + state
				city_2 = names[4][:-1] +  state       
				county_city_list = [county, city_1, city_2]   
			elif names[0] == 'Augusta,':
				county = names[0][:-1] + ' County' + state
				city_1 = names[1] + state
				city_2 = names[3][:-1] +  state       
				county_city_list = [county, city_1, city_2]    
			elif names[0] == 'Maui':
				county_1 = names[0] + ' County' + state
				county_2 = names[2][:-1] + ' County' + state
				county_city_list = [county_1, county_2]               
			else:
				county = names[0] + ' County' + state
				city = names[2][:-1] + state
				county_city_list = [county, city]
			msa_df.at[i,3] = county_city_list #[county, city]
			# print(i, msa_df.iloc[i,3])
		elif '(Independent City)' in msa_df.iloc[i,3]:
			msa_df.at[i,3] = msa_df.iloc[i,3].replace(' (Independent City),',",")
			# print(i, msa_df.iloc[i,3])
		else:
			msa_df.at[i,3] = msa_df.iloc[i,3].replace(',', " County,")
			# print(i, msa_df.iloc[i,3])
	msa_muni_df = pd.DataFrame(columns=['municipality'])
	msa_list = msa_df[1].unique()
	for msa in msa_list:
		msa_muni_df.at[msa, 'municipality'] = flatten(msa_df.loc[msa_df[1]==msa,3].to_list())
	msa_muni_df.index.name = 'msa'    
	msa_muni_df.index = msa_muni_df.index.str.replace(" (Metropolitan Statistical Area)","", regex=False)
	if save:
		msa_df.to_csv('../../data/tidy/metrolist.csv')
		msa_muni_df.to_csv('../../data/tidy/msa-municipality.csv')
	return msa_muni_df


def get_msa_network_stats(df, msa, save=True):	
	try:
		munis = df.loc[msa,'municipality']
		print ('Processing MSA: '+ msa + '; Municipalities: ', munis)		
		graph = ox.graph_from_place(munis, network_type='drive')            
		gdf = ox.geocode_to_gdf(munis) 
		msa_crs = gdf.estimate_utm_crs()
		msa_area = gdf.to_crs(crs=msa_crs).area.sum() #in square meters
		bldgs = ox.geometries_from_place(munis, tags={'building':True}) # Retrieve buildings from the area:
		bldgs = bldgs.iloc[bldgs.index.get_level_values('element_type')=='way']['geometry']
		bld_area = bldgs.to_crs(crs=msa_crs).area.sum()
		print ('calculating ' + msa +' stats')
		stats = ox.basic_stats(graph, area=msa_area) #, clean_int_tol=15)
		stats = pd.Series(stats)
		del stats['streets_per_node_counts']
		del stats['streets_per_node_proportions']            
		statsDF = pd.DataFrame(stats)
		statsDF=statsDF.T
		statsDF.insert(0, 'msa', [msa])
		statsDF['builing_area_sqkm'] = bld_area/1e6
		statsDF['area_sqkm'] = msa_area/1e6
		output_file = '../../data/tidy/msa-osm-stats.csv'
		statsDF.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
		if save:
			print ('saving ' + msa)                
			name= '../../data/tidy/graphs/' + msa + '.graphml'
			ox.save_graphml(graph, name)
		#fig, ax = ox.plot_graph(graph, node_size=0, edge_linewidth=0.2, show=False, save=True, filename=city, file_format='png')
	except Exception as e:
		print(e)
		print (msa + ' skipped')
	return


if __name__ == "__main__":
	start = time.time()
	df = get_msa_municipality_list_df(True)
	#ncpus = multiprocessing.cpu_count() - 1
	Parallel(n_jobs=2)(
		delayed(get_msa_network_stats)(df, i) for i in df.index
		)
	elapsed = (time.time() - start)
	print("Network stats completed in (h/m/s/ms):", str(timedelta(seconds=elapsed)))
