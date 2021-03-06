
package mcts

import (
	"fmt"
	"time"
	"math/rand"
	"math"
	"os"
)



type State interface {
	AvailableMovesMCTS() 	[]Move
	Copy() 				State
	GameOver()			bool
	Move(m Move)		State
	EvaluationScore()	float64

	IsEndTurn()			bool
	Print()
}

type Move interface {
	Probability()	float64
}

type Node struct {
	Id		 		int
	Parent   		*Node
	Children 		[]*Node
	State    		State
	Outcome     	float64
	Visits   		int
	ByMove			Move
	EndTurn  		bool
	UnexploreMoves	[]Move
}

func NewNode(parent *Node, state State, move Move) *Node {
	return &Node{
		Id:		  1,
		Parent:   parent,
		Children: nil,
		State:    state,
		Outcome:     0,
		Visits:   0,
		ByMove: move,
		EndTurn: false,
		UnexploreMoves: nil,
	}
}
func (n *Node) DeleteUnexploreMoves(m Move) (error) {
	idx := -1
	if n.UnexploreMoves == nil || len(n.UnexploreMoves) == 0 {
		return fmt.Errorf("Can't remove Move in empty list")
	}
	for i, tmp_m := range(n.UnexploreMoves) {
		if  tmp_m == m {
			idx = i
			break
		}
	}
	len_moves := len(n.UnexploreMoves)
	if idx != -1 && len_moves > 1 {
		n.UnexploreMoves = append(n.UnexploreMoves[:idx], n.UnexploreMoves[idx+1:]...)
	} else if idx != -1 {
		n.UnexploreMoves = make([]Move, 0)
	}
	return nil
}

func (n *Node) UpdateScore(score float64) {
	n.Visits++
	n.Outcome += score
}

func MonteCarloTimeout(state State, bias float64, iteration, simulation, timeout int) (*Node, error) {
	var node *Node
	var done chan bool = make(chan bool)

	now := time.Now()
	to	:= time.Nanosecond * time.Duration(timeout) * (1000000)
	time.AfterFunc(to, func () {
		done <- true
	})
	//defer timer.Close()
	var running = true
	var root_node *Node

	root_node = NewNode(nil, state, nil)

	for ; running ; {

		node = MCSelection(root_node, bias)
		node = MCExpansion(node)

		score := MCSimulation(node.State, simulation)
		MCBackPropagation(node, score)

		select {
		case <-done:
			fmt.Fprintln(os.Stderr, "Timeout", time.Since(now))
			running = false
			break
		default:
		}
	}
	return root_node, nil
}
func MonteCarlo(root_node *Node, bias float64, iteration, simulation int) (*Node, error) {
	var node *Node

	if iteration <= 0 { return nil, fmt.Errorf("Iteration should be > 0") }
	if simulation < -1 { return nil, fmt.Errorf("Simulation should be > 0") }

	for i := 0; i < iteration; i++ {

		node = MCSelection(root_node, bias)
		node = MCExpansion(node)

		if len(root_node.UnexploreMoves) == 0 && len(root_node.Children) == 1 {
			return node, nil
		}

		score := MCSimulation(node.State, simulation)
		MCBackPropagation(node, score)
	}

	return node, nil
}

func MCSelection(node *Node, bias float64) *Node {
	var candidate_node *Node

	if node.UnexploreMoves == nil {
		node.UnexploreMoves = node.State.AvailableMovesMCTS()
	}


	if len(node.UnexploreMoves) == 0 && node.Children != nil && len(node.Children) > 0 {
		candidate_node = nil
		score := -1.0
		for _, n := range node.Children {
			child_score := MCCalculateScore(n, bias)
			if child_score > score || candidate_node == nil {
				score = child_score
				candidate_node = n
			}
		}
		if candidate_node == nil {
			return node
		}
		return MCSelection(candidate_node, bias)
	}
	return node
}
func MCCalculateScore(node *Node, bias float64) float64 {
	if node.Parent == nil {
		return 0
	}
	exploitScore := float64(node.Outcome) / float64(node.Visits)
	exploreScore := math.Sqrt(2 * math.Log(float64(node.Parent.Visits)) / float64(node.Visits))
	exploreScore = bias * exploreScore

	return exploitScore + exploreScore
}
func MCExpansion(node *Node) *Node {

	if len(node.UnexploreMoves) == 0 {
		return node
	}


	new_state := node.State.Copy()

	// Pick random move
	source 	:= rand.NewSource(time.Now().UnixNano())
	random 	:= rand.New(source)
	rmove 	:= node.UnexploreMoves[random.Intn(len(node.UnexploreMoves))]

	new_state.Move(rmove)
	new_node := NewNode(node, new_state, rmove)
	node.DeleteUnexploreMoves(rmove)

	new_node.Parent = node
	node.Children = append(node.Children, new_node)

	return new_node
}
func MCSimulation(state State, simulation int) float64 {

	var moves []Move = nil


	source 	:= rand.NewSource(time.Now().UnixNano())
	random 	:= rand.New(source)

	simulate_state := state.Copy()

	for i := 0 ; ! simulate_state.GameOver()  && (simulation == -1 || i < simulation) ; i++  {

		moves = simulate_state.AvailableMovesMCTS()
		if moves != nil || len(moves) == 0 {
			break
		}
		move := moves[random.Intn(len(moves))]
		simulate_state.Move(move)
	}
	return simulate_state.EvaluationScore()
}

func MCBackPropagation(node *Node, score float64) *Node {

	for node.Parent != nil {
		node.UpdateScore(score)
	}

	node.Visits++
	return node
}
