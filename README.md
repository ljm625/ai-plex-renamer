# Plex Renamer

优先使用本地免费的 `guessit` 解析媒体文件名，并把电影、电视剧文件重命名为 Plex 更容易识别的格式。

如果配置了 TMDB API，会在 GuessIt 之后查询标准片名和发行年份，尽量生成带年份的 Plex 文件名，减少同名作品匹配错误。当本地解析无法判断出可用的季/集/电影信息时，才会调用 NVIDIA AI endpoint 作为 AI 兜底。

默认只预览，不会真的改名；加 `--apply` 才会执行重命名。

## 安装

```bash
python -m pip install -e .
```

`guessit` 是本地解析，不需要 API key 或网络。TMDB 是可选增强，需要配置 `TMDB_BEARER_TOKEN` 或 `TMDB_API_KEY`。NVIDIA AI endpoint 仅在本地解析失败或结果置信度较低时使用，需要配置 `NVIDIA_API_KEY`。

## 使用

处理顺序：

```text
GuessIt 本地解析
GuessIt 不够时用内置规则兜底
对可用结果用 TMDB 补标准标题/年份
仍无可用结果时用 NVIDIA AI endpoint 兜底
```

如果文件位于作品名文件夹内，并且文件名包含 `前編`、`後編`、`第1話`、`Part 1` 这类分集标记，会优先按剧集处理

同一目录下的多个文件会作为一组处理，文件名包含 `#1`、`＃2`、`第3話` 等序号时会互相参考并按剧集命名


如果这组文件仍然需要 AI 兜底，会把同目录文件一起发给 NVIDIA AI endpoint，让模型参考整组上下文，而不是逐个孤立判断。

启用 TMDB：

```bash
export TMDB_BEARER_TOKEN="你的 TMDB API Read Access Token"
ai-plex-renamer "/path/to/media"
```

也可以用 v3 API key：

```bash
export TMDB_API_KEY="你的 TMDB v3 API key"
ai-plex-renamer "/path/to/media"
```

不使用 TMDB：

```bash
ai-plex-renamer "/path/to/media" --no-tmdb
```

TMDB 署名说明：This product uses the TMDB API but is not endorsed or certified by TMDB.

启用 NVIDIA AI 兜底：

```bash
export NVIDIA_API_KEY="你的 NVIDIA API key"
ai-plex-renamer "/path/to/media"
```

可选指定模型和 endpoint：

```bash
ai-plex-renamer "/path/to/media" \
  --nvidia-model meta/llama-3.1-8b-instruct \
  --nvidia-base-url https://integrate.api.nvidia.com/v1
```

预览指定文件夹：

```bash
ai-plex-renamer "/path/to/media"
```

确认执行重命名：

```bash
ai-plex-renamer "/path/to/media" --apply
```

指定媒体库类型：

```bash
ai-plex-renamer "/path/to/tv" --type tv --apply
ai-plex-renamer "/path/to/movies" --type movie --apply
```

常用选项：

```bash
ai-plex-renamer "/path/to/media" \
  --type auto \
  --tmdb-language en-US \
  --tmdb-include-adult \
  --nvidia-model meta/llama-3.1-8b-instruct \
  --collision skip \
  --apply
```

如果只想用本地解析，不调用 NVIDIA AI：

```bash
ai-plex-renamer "/path/to/media" --no-ai
```

查看详细处理过程：

```bash
ai-plex-renamer "/path/to/media" --verbose
```

`--verbose` 会把分组、本地解析、TMDB 请求/响应、NVIDIA 请求/响应打印到 stderr；`Authorization`、`api_key` 等敏感信息会自动脱敏。

## Plex 命名结果

电视剧：

```text
Show Name (2023) - S01E02 - Episode Title.mkv
```

电影：

```text
Movie Name (2024).mkv
```

无法可靠判断的文件会被跳过，不会强行改成错误名字。
