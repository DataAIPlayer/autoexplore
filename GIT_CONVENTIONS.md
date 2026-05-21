# Git 管理规范

## 分支策略

### 主分支
- **main**: 生产环境分支，始终保持可部署状态

### 开发分支
- **develop**: 开发主分支，所有开发合并到此分支
- **feature/***: 功能开发分支，从 develop 分出，完成后合并回 develop
- **bugfix/***: 问题修复分支，从 develop 分出，完成后合并回 develop
- **hotfix/***: 紧急修复分支，从 main 分出，完成后合并回 main 和 develop

### 分支命名规范
```
feature/<feature-name>    # 新功能开发
bugfix/<bug-description>   # 问题修复
hotfix/<hotfix-description> # 紧急修复
```

## Commit 规范

### Commit Message 格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型
- **feat**: 新功能
- **fix**: 问题修复
- **docs**: 文档变更
- **style**: 代码格式变更（不影响功能）
- **refactor**: 重构（既不是新功能也不是修复）
- **perf**: 性能优化
- **test**: 测试相关
- **chore**: 构建/工具链相关

### Subject 规范
- 使用祈使句，如 "add" 而非 "added" 或 "adds"
- 首字母小写
- 结尾不加句号
- 限制在 50 字符以内

### Body 规范
- 说明 "what" 和 "why"，而非 "how"
- 每行限制在 72 字符以内

### Footer 规范
- 关联 Issue: `Closes #123` 或 `Refs #123`
- 破坏性变更: `BREAKING CHANGE:`

### 示例

```
feat(auth): add user registration

Add email-based user registration with password validation.
Includes email verification flow.

Closes #42
```

```
fix(instance): resolve container startup failure

Fix docker compose configuration issue that prevented
instances from starting properly.

Fixes #87
```

## 工作流程

### 功能开发流程
1. 从 `develop` 创建功能分支
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/user-registration
   ```
2. 开发并提交
3. 推送到远程
   ```bash
   git push origin feature/user-registration
   ```
4. 创建 Pull Request 到 `develop`
5. 代码审查通过后合并
6. 删除功能分支

### 问题修复流程
1. 从 `develop` 创建修复分支
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b bugfix/login-error
   ```
2. 修复并提交
3. 推送并创建 PR
4. 审查通过后合并
5. 删除修复分支

### 紧急修复流程
1. 从 `main` 创建修复分支
   ```bash
   git checkout main
   git pull origin main
   git checkout -b hotfix/critical-security-fix
   ```
2. 修复并提交
3. 推送并创建 PR 到 `main`
4. 审查通过后合并到 `main`
5. 将修复合并回 `develop`
6. 打标签发布
7. 删除修复分支

## 发布流程

### 版本号规范
遵循 [语义化版本](https://semver.org/lang/zh-CN/)
- 主版本号.次版本号.修订号 (如 1.2.3)

### 发布步骤
1. 更新版本号
2. 合并 `develop` 到 `main`
3. 创建标签
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```
4. 部署到生产环境

## 代码审查

### PR 要求
- 清晰描述变更内容和原因
- 关联相关 Issue
- 通过所有自动化检查
- 至少一人审查通过

### 审查要点
- 代码质量和可读性
- 遵循项目规范
- 测试覆盖充分
- 无明显性能问题

## 合并策略

- **feature/bugfix**: Squash and merge（合并成一个 commit）
- **hotfix**: Merge commit（保留历史）
- **develop → main**: Merge commit

## 其他规范

### Rebase 禁止
- 禁止对已推送的 commit 进行 rebase
- 使用 merge 来整合变更

### Commit 频率
- 小而频繁的 commit 优于大而稀疏的
- 每个 commit 完成一个逻辑单元

### 冲突解决
- 及时解决冲突
- 保持提交历史清晰
