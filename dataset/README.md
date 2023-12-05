# identity-aligned cross-site UGC dataset

This dataset contains two aligned social networks, covering Chinese and English websites. One is between Weibo and Douban, and the other is between Instagram and Twitter.

Since the original dataset involves users' privacy issues, we only upload the training dataset after preprocessing. If you need the original dataset, please contact me ([q_zhou21@m.fudan.edu.cn](mailto:q_zhou21@m.fudan.edu)). I will reply you as soon as possible.

## Weibo-Douban Dataset

### Original corpus

The dataset collects posts of 5,323 Weibo users and 7,202 Douban users during the three years from 2020.11.1 to 2023.10.30. We totally obtain 4,064 pairs of anchor users through their publicly available account information. 

### Data process

1. For the topics discussed by users on these two platforms, we drop the topics discussed less than ten times and those discussed by less than three users to form the union topic set. Thus, we get 6,382 topics on Weibo and 547 topics on Douban.
2. For every post, we use a pre-trained Glove vector (50 dimensions) to represent each word and set the average of word vectors as the representation of the post.
3. We process the samples by median-split, i.e., setting the intermediate time point of one year as the boundary. The task is to predict what topics a user will be interested in after the boundary, given her/his posting history before the boundary. For example, for a user having published posts during 2020.11.1-2021.10.31, the boundary is 2021.5.1. The posts published before 2021.5.1 are utilized as the history, and the $t$ hashtag topics appearing in the posts published after 2021.5.1 are set as the prediction labels, constituting a collection of $t$ positive samples. At the same time, we randomly sample $t$ topics from the topic set that the user has not participated in to generate $t$ negative samples.
4. Finally, we obtain the cross-site dataset containing 37,408 samples for cross-site CPP task.

### Statistics

| Platform | #Users | #User Posts | #Topics | #Topic Posts | #Anchor Users | #Union Topics | #CPP Tsak Samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Weibo | 5,323 | 601,882 | 6,382 | 293,347 | 4,064 | 6,875 | 37,408 |
| Douban | 7,202 | 1,617,071 | 547 | 28,533 |  |  |  |

### Format

- Weibo-Douban.npy: Organized in Key-Value format, each Key-Value corresponds to a sample.
    - wb_user_posts: user's posting history on the Weibo platform
    - wb_user_topics:  user's topic participation history on the Weibo platform
    - db_user_posts: user's posting history on the Douban platform
    - db_user_topics: user's topic participation history on the Douban platform
    - topic: the topic to be predict whether users are interested
    - topic_posts: posts related to the predicting topic
    - y_label: prediction task label

## Instagram-Twitter Dataset

### Original corpus

This dataset is based on the original dataset created by Chen et al.[2]. The dataset collects posts of 9,238 Instagram users and 7,321 Twitter users from 2007.4.1-2018.11.30. We totally obtain 6,652 pairs of anchor users through the matching relationships of users collated by Lim et al.[1]. 

### Data process

1. For the topics discussed by users on these two platforms, we drop the topics discussed less than one hundred times and those discussed by less than ten users to form the union topic set. Thus, we get 3,998 topics on Instagram and 485 topics on Twitter.
2. For every post, we use a pre-trained Glove vector (50 dimensions) to represent each word and set the average of word vectors as the representation of the post.
3. We process the samples by median-split, i.e., setting half of the total number of user tweets as the boundary. The task is to predict what topics a user will be interested in after the boundary, given her/his posting history before the boundary. For example, for a user having published one hundred posts during, the boundary is the fiftyth post(the posts are ordered by published time). The posts published before the fiftyth post are utilized as the history, and the $t$ hashtag topics appearing in the posts published after the fiftyth post are set as the prediction labels, constituting a collection of $t$ positive samples. At the same time, we randomly sample $t$ topics from the topic set that the user has not participated in to generate $t$ negative samples.
4. Finally, we get 349,439 samples, and randomly select 35,000 samples (10\%) for the cross-site dataset.

### Statistics

| Platform | #Users | #User Posts | #Topics | #Topic Posts | #Anchor Users | #Union Topics | #CPP Tsak Samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Instagram | 9,238 | 1,143,520 | 3,998 | 1,666,024 | 6,652 | 4,123 | 35,000 |
| Twitter | 7,321 | 1,229,842 | 485 | 128,157 |  |  |  |

### Format

- Instagram-Twitter.npy: Organized in Key-Value format, each Key-Value corresponds to a sample.
    - ins_user_posts: user's posting history on the Instagram platform
    - ins_user_topics:  user's topic participation history on the Instagram platform
    - tw_user_posts: user's posting history on the Tweitter platform
    - tw_user_topics: user's topic participation history on the Tweitter platform
    - topic: the topic to be predict whether users are interested
    - topic_posts: posts related to the predicting topic
    - y_label: prediction task label

## Reference

[1] Lim, Bang Hui , et al. "#mytweet via Instagram: Exploring User Behaviour across Multiple Social Networks." *IEEE* (2015).

[2] Chen, Xiaolin , et al. "User Identity Linkage across Social Media via Attentive Time-aware User Modeling." *IEEE Transactions on Multimedia* PP.99(2020):1-1.
