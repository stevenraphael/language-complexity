When testing ANN models on various executive function tasks, there is a clear separation in areas of activation between language and non-language tasks. While this may imply separation of language and non-language tasks in human brains as well, one possible confound for this result is the complexity of the tasks, mainly that language tasks are more complex than the other non-language tasks tested on the models. Thus, the purpose of this project is to find ways of reducing the complexity of
language tasks to see if this separation is maintained.  The main goal was to find ways of reducing the complexity of language data used to train and test models. These required two things: one, to find sets of more implicit English language data, and two, to determine features for measuring text difficulty. This package is the code produced alongside this project.

complexity_measures.py: All functions with language complexity measurements

scraping.py : Used for scraping sites, designed with wikipedia and simple wikipedia in mind

generate_fake.py: Sample of PCFG generation
