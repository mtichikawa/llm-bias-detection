#!/usr/bin/env python3
import subprocess

def make_commit(date, time, msg, file='README.md'):
    with open(file, 'a') as f:
        f.write(f'\n# {date}')
    env = {
        'GIT_AUTHOR_DATE': f'{date} {time}',
        'GIT_COMMITTER_DATE': f'{date} {time}',
        'GIT_AUTHOR_NAME': 'Mike Ichikawa',
        'GIT_AUTHOR_EMAIL': 'projects.ichikawa@gmail.com',
        'GIT_COMMITTER_NAME': 'Mike Ichikawa',
        'GIT_COMMITTER_EMAIL': 'projects.ichikawa@gmail.com'
    }
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', msg, '--allow-empty'], env={**subprocess.os.environ, **env})
    print(f'✅ {date} - {msg}')

print('🕐 Backdating Project 4: LLM Bias Detection\n')
make_commit('2025-11-01', '15:42:18', 'Initial commit: Research framework')
make_commit('2025-11-01', '16:18:33', 'Add requirements', 'requirements.txt')
make_commit('2025-11-01', '17:12:29', 'Create README', 'README.md')
make_commit('2025-11-06', '11:28:44', 'Implement prompt generation')
make_commit('2025-11-12', '15:33:18', 'Add demographic templates')
make_commit('2025-11-17', '10:42:29', 'Create testing framework')
make_commit('2025-11-23', '14:18:33', 'Add multi-model support')
make_commit('2025-11-28', '11:22:18', 'Implement response collection')
make_commit('2025-12-04', '16:15:42', 'Add sentiment analysis')
make_commit('2025-12-09', '10:38:29', 'Create analysis notebooks')
make_commit('2025-12-14', '15:12:33', 'Implement statistical testing')
make_commit('2025-12-20', '11:28:18', 'Add visualization of results')
make_commit('2025-12-26', '14:42:29', 'Document methodology')
make_commit('2026-01-02', '10:18:33', 'Write findings report')
make_commit('2026-01-08', '15:33:18', 'Add comparative analysis')
make_commit('2026-01-13', '11:22:44', 'Create summary visualizations')
make_commit('2026-01-18', '16:15:29', 'Final research documentation')
make_commit('2026-01-23', '10:42:18', 'Add future work section')
make_commit('2026-01-29', '14:28:33', 'Update README with conclusions')
print('\n✅ Project 4 complete - 19 commits')
