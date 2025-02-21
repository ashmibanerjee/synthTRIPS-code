# SynthTRIPS: A Knowledge-Grounded Framework for Benchmark Query Generation for Personalized Tourism Recommenders

This repository contains the code files for the SynthTRIPS Query Generation Framework. SynthTRIPS is a novel framework for generating synthetic travel queries using LLMs grounded in curated knowledge base (KB). Our approach combines persona-based preferences (e.g., budget, travel style) with explicit sustainability filters (e.g., walkability, air quality) to produce realistic and diverse queries. 

We mitigate hallucination and ensure factual correctness by grounding the LLM responses in the KB. We formalize the query generation process and introduce evaluation metrics for assessing realism and alignment. Both human expert evaluations and automatic LLM-based assessments demonstrate the effectiveness of our synthetic dataset in capturing complex personalization aspects underrepresented in existing datasets. While our framework was developed and tested for personalized city trip recommendations, the methodology applies to other recommender system domains.

> The pipeline is availabe to test on Colab: 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashmibanerjee/synthTRIPS-code/blob/main/SynthTRIPS_Query_Gen_Pipeline.ipynb)

## Run 

To execute the pipeline and/or the tests, please follow the steps below: 

1. Create a subfolder under root called `data/` and clone the [dataset](https://huggingface.co/datasets/ashmib/synthTRIPS) from HuggingFace there. 
2. Install the requirements: `pip install -r requirements.txt` 


## Acknowledgements
We thank the Google AI/ML Developer Programs team for supporting us with Google Cloud Credits.


## Citation 

If you use the dataset or framework, please cite the following: 

    @misc{banerjee2025synthTRIPS,
        title={SynthTRIPS: A Knowledge-Grounded Framework for Benchmark Query Generation for Personalized Tourism Recommenders},
        author={Ashmi Banerjee and Adithi Satish and Fitri Nur Aisyah and
        Wolfgang Wörndl and Yashar Deldjoo},
        year={2025},
        eprint={},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
