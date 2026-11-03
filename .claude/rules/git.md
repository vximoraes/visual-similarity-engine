---
description: Fluxo de git. Sempre ativo.
---

# Git

## Sempre pedir antes

- **Nunca rode `git commit`, `git push`, `git merge`, `git rebase` ou `git tag` sem permissão explícita do
  usuário para aquela ação específica.** Permissão anterior não vale para a próxima: pergunte de novo.
- Antes de começar uma tarefa nova, pergunte se o usuário quer uma branch dedicada (`feat/...`, `fix/...`).

## Commits: Conventional Commits, sem escopo

`<tipo>: <descrição>`, nunca `<tipo>(escopo): ...`. Tipos: `feat`, `fix`, `chore`, `refactor`, `style`,
`docs`, `test`. Descrição em português. O corpo explica o porquê quando não for óbvio.

## Branches e merge

- Nome curto com prefixo do tipo, criada a partir da `main`.
- Merge sempre `git merge --no-ff <branch>`; branch nunca é apagada (local ou remota).
- Force-push, `reset --hard`, `clean -f` e apagar branch são bloqueados pelo hook `.claude/hooks/validate-bash.sh`.
