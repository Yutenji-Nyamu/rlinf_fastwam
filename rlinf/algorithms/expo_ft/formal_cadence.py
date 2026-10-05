"""CPU-only boundary accounting for the formal EXPO physical-action budget."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FormalCadence:
    max_physical_actions: int
    warmup_episodes: int = 10
    actions_per_call: int = 40
    max_episode_actions: int = 200
    minimum_online_actions: int = 64
    prior_budget_truncations: int = 0
    episodes_completed: int = 0
    physical_actions: int = 0
    warmup_actions: int = 0
    post_warmup_actions: int = 0
    carry_actions: int = 0
    pending_calls: int = 0
    completed_calls: int = 0
    budget_truncated_episodes: int = 0

    VERSION = 2
    COUNTERS = ('episodes_completed', 'physical_actions', 'warmup_actions',
                'post_warmup_actions', 'carry_actions', 'pending_calls',
                'completed_calls', 'budget_truncated_episodes')

    def __post_init__(self):
        for name in ('max_physical_actions', 'actions_per_call', 'max_episode_actions',
                     'minimum_online_actions'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if type(self.warmup_episodes) is not int or self.warmup_episodes < 0:
            raise ValueError('warmup_episodes must be a nonnegative integer')
        if type(self.prior_budget_truncations) is not int or self.prior_budget_truncations < 0:
            raise ValueError('prior_budget_truncations must be a nonnegative integer')
        self._validate()

    @property
    def remaining_actions(self):
        return self.max_physical_actions - self.physical_actions

    @property
    def in_warmup(self):
        return self.episodes_completed < self.warmup_episodes

    @property
    def can_learn(self):
        return not self.in_warmup and self.physical_actions >= self.minimum_online_actions

    def _validate(self):
        for name in self.COUNTERS:
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError('Invalid cadence counter: ' + name)
        warmup_count = min(self.episodes_completed, self.warmup_episodes)
        online_count = self.episodes_completed - warmup_count
        if (self.physical_actions != self.warmup_actions + self.post_warmup_actions or
                not 0 <= self.physical_actions <= self.max_physical_actions or
                not warmup_count <= self.warmup_actions <= warmup_count * self.max_episode_actions or
                not online_count <= self.post_warmup_actions <= online_count * self.max_episode_actions or
                self.carry_actions != self.post_warmup_actions % self.actions_per_call or
                self.completed_calls + self.pending_calls != self.post_warmup_actions // self.actions_per_call or
                (self.completed_calls and not self.can_learn) or
                self.budget_truncated_episodes > min(self.prior_budget_truncations + 1, self.episodes_completed) or
                (self.budget_truncated_episodes > self.prior_budget_truncations and self.remaining_actions != 0)):
            raise ValueError('Cadence action/episode/debt invariants differ')

    def finish_episode(self, actions: int, *, budget_truncated: bool = False) -> int:
        """Publish one real episode boundary; warmup actions never create debt."""
        if self.pending_calls and self.can_learn:
            raise ValueError('Drain pending boundary calls before another rollout')
        if (type(actions) is not int or not 1 <= actions <= self.max_episode_actions or
                actions > self.remaining_actions):
            raise ValueError('Episode physical actions exceed the inherited limit/budget')
        if type(budget_truncated) is not bool or (budget_truncated and actions != self.remaining_actions):
            raise ValueError('Budget truncation must consume exactly the remaining real-action budget')
        if self.in_warmup:
            self.warmup_actions += actions
        else:
            self.post_warmup_actions += actions
            new_calls, self.carry_actions = divmod(self.carry_actions + actions, self.actions_per_call)
            self.pending_calls += new_calls
        self.physical_actions += actions
        self.episodes_completed += 1
        self.budget_truncated_episodes += int(budget_truncated)
        self._validate()
        return self.pending_calls

    def complete_call(self):
        """Commit one *completed* Q20→FM→editor→temperature call."""
        if self.pending_calls < 1 or not self.can_learn:
            raise ValueError('No eligible scheduled call; warmup/minimum/carry cannot be trained')
        self.pending_calls -= 1
        self.completed_calls += 1
        self._validate()

    def state_dict(self):
        self._validate()
        state = {'version': self.VERSION,
                'contract': {name: getattr(self, name) for name in (
                    'max_physical_actions', 'warmup_episodes', 'actions_per_call', 'max_episode_actions',
                    'minimum_online_actions')},
                'counters': {name: getattr(self, name) for name in self.COUNTERS}}
        if self.prior_budget_truncations:
            state['contract']['prior_budget_truncations'] = self.prior_budget_truncations
        return state

    def load_state_dict(self, state):
        if (set(state) != {'version', 'contract', 'counters'} or
                state['version'] != self.VERSION or state['contract'] != self.state_dict()['contract'] or
                set(state['counters']) != set(self.COUNTERS)):
            raise ValueError('Formal cadence budget/config/schema differs; strict restore refused')
        candidate = FormalCadence(**state['contract'], **state['counters'])
        for name in self.COUNTERS:
            setattr(self, name, getattr(candidate, name))
