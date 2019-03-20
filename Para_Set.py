import yaml

Para={}
Para['version'] = 'V1'
Para['dataset'] = 'MQ2007' #'MSLR-WEB10K'
Para['model'] = 'Softmax_10'
Para['Nfeature'] = 46  #136
Para['Learningrate'] = 0.0001
Para['Nepisode'] = 100
Para['Lenepisode'] = 100

Para_file = open('Para_info.yml', 'w+')
yaml.dump(Para, Para_file)

Para = yaml.load(open('Para_info.yml', 'r'))
print(Para)