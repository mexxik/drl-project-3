## About

This is a solution for Project 3 (Collaboration and Competition) of Udacity Deep Reinforcement Learning Nanodegree Program

The project is implemented in Python 3.5 and PyTorch 1.0.1.

## Environment Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## Installation

* Install dependencies:
`pip install -r dependencies.txt`

* Get Unity Reacher Environment for your OS and put it in the root directory of this projects:
    * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    * [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    * [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    * [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Structure

All classes and utility function are implemented in `Tennis.ipynb` note book.

## Training

`Tennis.ipynb` contains all the necessary code to train the models. Some 

## Report

Report is inside `Report.md`.
